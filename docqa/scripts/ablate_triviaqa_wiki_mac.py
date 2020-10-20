import argparse
from datetime import datetime
from typing import Optional

from tensorflow.keras.initializers import TruncatedNormal

from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ContextLenBucketedKey
from docqa.dataset import ClusteredBatcher
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from docqa.text_preprocessor import WithIndicators, TextPreprocessor
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWikiDataset
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion

from docqa.doc_qa_models import MacAttention
from docqa.encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder, GroupedSpanAnswerEncoder
from docqa.nn.attention import BiAttention, AttentionEncoder, StaticAttentionSelf
from docqa.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from docqa.nn.layers import NullBiMapper, SequenceMapperSeq, Conv1d, FullyConnected, \
    ChainBiMapper, ConcatWithProduct, ResidualLayer, VariationalDropoutLayer, MaxPool
from docqa.nn.recurrent_layers import CudnnGru
from docqa.nn.similarity_layers import TriLinear
from docqa.nn.span_prediction import ConfidencePredictor, BoundsPredictor, IndependentBoundsGrouped, \
    IndependentBoundsSigmoidLoss
from docqa.nn.mac import MacNetwork

def get_model(char_th: int, dim: int, mode: str, preprocess: Optional[TextPreprocessor]):
    recurrent_layer = CudnnGru(dim, w_init=TruncatedNormal(stddev=0.05))

    if mode.startswith("shared-norm"):
        answer_encoder = GroupedSpanAnswerEncoder()
        predictor = BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            ),
            span_predictor=IndependentBoundsGrouped(aggregate="sum")
        )
    elif mode == "confidence":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = ConfidencePredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer,
            ),
            AttentionEncoder(),
            FullyConnected(80, activation="tanh"),
            aggregate="sum"
        )
    elif mode == "sigmoid":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            ),
            span_predictor=IndependentBoundsSigmoidLoss()
        )
    elif mode == "paragraph" or mode == "merge":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = BoundsPredictor(ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ))
    else:
        raise NotImplementedError(mode)

    return MacAttention(
        encoder=DocumentAndQuestionEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=char_th, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        preprocess=preprocess,
        word_embed_layer=None,
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        mac=MacNetwork(2, dim*2),
        match_encoder=SequenceMapperSeq(FullyConnected(dim * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(0.8),
                                            recurrent_layer,
                                            VariationalDropoutLayer(0.8),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(dim * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(0.8)),
        predictor=predictor
    )

def main():
    parser = argparse.ArgumentParser(description='Train a model on TriviaQA wiki')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    # Note I haven't tested modes other than `shared-norm` on this corpus, so
    # some things might need adjusting
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("-t", "--n_tokens", default=400, type=int,
                        help="Paragraph size")
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with"
                        )
    args = parser.parse_args()
    mode = args.mode

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    model = get_model(100, 140, mode, WithIndicators())

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(args.n_tokens),
                                                ShallowOpenWebRanker(16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge", per_doc=False)]
    oversample = [1] * 2  # Sample the top two answer-containing paragraphs twice

    if mode == "paragraph":
        n_epochs = 120
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True),
                                          oversample,  only_answers=True)
    elif mode == "confidence" or mode == "sigmoid":
        if mode == "sigmoid":
            n_epochs = 640
        else:
            n_epochs = 160
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), oversample)
    else:
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)
        train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)

    data = TriviaQaWikiDataset()

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=10, log_period=30, eval_period=1800, save_period=1800,
        best_weights=("dev", "b8/question-text-f1"),
        eval_samples=dict(dev=None, train=6000)
    )

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(args.n_processes, 1000)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "Mode: " + args.mode + "\n" + notes

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()