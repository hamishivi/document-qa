import argparse
import json
from os import walk, mkdir, makedirs, remove
from os.path import relpath, join, exists
import pickle
from typing import Set
from collections import Counter

import numpy as np
import resource

import sys

import unicodedata

import config
from utils import group, split, flatten_iterable
from tqdm import tqdm
import re
from config import CORPUS_DIR
from data_processing.text_utils import NltkPlusStopWords, NltkAndPunctTokenizer

"""
Build a cache a tokenized version of the evidence corpus
"""


def _gather_files(input_root, output_dir, skip_dirs):
    if not exists(output_dir):
        mkdir(output_dir)

    all_files = []
    for root, dirs, filenames in walk(input_root):
        if skip_dirs:
            output = join(output_dir, relpath(root, input_root))
            if exists(output):
                continue
        path = relpath(root, input_root)
        if not exists(join(output_dir, path)):
            mkdir(join(output_dir, path))
        all_files += [join(path, x) for x in filenames]
    return all_files


def build_tokenized_corpus(input_root, tokenizer, output_dir, skip_dirs=False, n_processes=1):
    if not exists(output_dir):
        mkdir(output_dir)

    all_files = _gather_files(input_root, output_dir, skip_dirs)

    if n_processes == 1:
        voc = build_tokenized_files(all_files, input_root, output_dir, tokenizer)
    else:
        voc = set()
        from multiprocessing import Pool
        with Pool(n_processes) as pool:
            chunks = split(all_files, n_processes)
            chunks = flatten_iterable(group(c, 500) for c in chunks)
            pbar = tqdm(total=len(chunks), ncols=80)
            for v in pool.imap_unordered(_build_tokenized_files_t,
                                         [[c, input_root, output_dir, tokenizer] for c in chunks]):
                voc.update(v)
                pbar.update(1)
            pbar.close()

    voc_file = join(output_dir, "vocab.txt", "w")
    with open(voc_file, "w") as f:
        for word in sorted(voc):
            f.write(word)
            f.write("\n")


def build_tokenized_files(filenames, input_root, output_root, tokenizer) -> Set[str]:
    voc = set()
    for filename in filenames:
        out_file = filename[:filename.rfind(".")] + ".txt"
        with open(join(input_root, filename), "r") as in_file:
            text = in_file.read().strip()
        paras = [x for x in text.split("\n") if len(x) > 0]
        paragraphs = [tokenizer.tokenize_paragraph(x) for x in paras]

        for para in paragraphs:
            for i, sent in enumerate(para):
                voc.update(sent)

        with open(join(output_root, out_file), "w") as in_file:
            in_file.write("\n\n".join("\n".join(" ".join(sent) for sent in para) for para in paragraphs))
    return voc


def _build_tokenized_files_t(arg):
    return build_tokenized_files(*arg)


def extract_voc(corpus, doc_ids):
    voc = Counter()
    for i, doc in enumerate(doc_ids):
        voc.update(corpus.get_document(doc, flat=True))
    return voc


def _extract_voc_tuple(x):
    return extract_voc(*x)


def get_evidence_voc(corpus, n_processes=1):
    doc_ids = corpus.list_documents()
    voc = Counter()

    if n_processes == 1:
        for doc in tqdm(doc_ids):
            voc = corpus.get_document(doc, flat=True)
    else:
        from multiprocessing import Pool
        chunks = split(doc_ids, n_processes)
        chunks = flatten_iterable(group(x, 10000) for x in chunks)
        pbar = tqdm(total=len(chunks), ncols=80)
        with Pool(n_processes) as pool:
            for v in pool.imap_unordered(_extract_voc_tuple, [[corpus, c] for c in chunks]):
                voc += v
                pbar.update(1)
        pbar.close()

    return voc


def build_evidence_voc(corpus, override, n_processes):
    target_file = join(corpus.directory, "vocab.txt")
    if exists(target_file) and not override:
        raise ValueError()
    voc = get_evidence_voc(TriviaQaEvidenceCorpusTxt(), n_processes=n_processes).keys()
    with open(target_file, "w") as f:
        for word in sorted(voc):
            f.write(word)
            f.write("\n")


class TriviaQaEvidenceCorpusTxt(object):
    """ Corpus of the tokenized text from the given TriviaQa evidence documents """
    _split_all = re.compile("[\n ]")
    _split_para = re.compile("\n\n+")  # FIXME we should not have saved document w/extra spaces...

    def __init__(self, file_id_map=None):
        self.directory = join(CORPUS_DIR, "triviaqa/evidence")
        self.file_id_map = file_id_map

    def get_vocab(self):
        with open(join(self.directory, "vocab.txt"), "r") as f:
            return {x.strip() for x in f}

    def load_word_vectors(self, vec_name):
        filename = join(self.directory, vec_name + "_pruned.pkl")
        if exists(filename):
            with open(filename, "rb"):
                return pickle.load(filename)
        else:
            return None

    def list_documents(self):
        if self.file_id_map is not None:
            return list(self.file_id_map.keys())
        f = []
        for dirpath, dirnames, filenames in walk(self.directory):
            if dirpath == self.directory:
                # Exclude files in the top level dir, like the vocab file
                continue
            if self.directory != dirpath:
                rel_path = relpath(dirpath, self.directory)
                f += [join(rel_path, x[:-4]) for x in filenames]
            else:
                f += [x[:-4] for x in filenames]
        return f

    def get_document(self, doc_id, n_tokens=None, flat=False):
        if self.file_id_map is None:
            file_id = doc_id
        else:
            file_id = self.file_id_map.get(doc_id)

        if file_id is None:
            return None

        file_id = join(self.directory, file_id + ".txt")
        if not exists(file_id):
            print(file_id)
            return None

        with open(file_id, "r") as f:
            if n_tokens is None:
                text = f.read()
                if flat:
                    return [x for x in self._split_all.split(text) if len(x) > 0]
                else:
                    paragraphs = []
                    for para in self._split_para.split(text):
                        paragraphs.append([sent.split(" ") for sent in para.split("\n")])
                    return paragraphs
            else:
                paragraphs = []
                paragraph = []
                cur_tokens = 0
                for line in f:
                    if line == "\n":
                        if not flat and len(paragraph) > 0:
                            paragraphs.append(paragraph)
                            paragraph = []
                    else:
                        sent = line.split(" ")
                        sent[-1] = sent[-1].rstrip()
                        if len(sent) + cur_tokens > n_tokens:
                            if n_tokens != cur_tokens:
                                paragraph.append(sent[:n_tokens-cur_tokens])
                            break
                        else:
                            paragraph.append(sent)
                            cur_tokens += len(sent)
                if flat:
                    return flatten_iterable(paragraph)
                else:
                    if len(paragraph) > 0:
                        paragraphs.append(paragraph)
                    return paragraphs


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-o", "--output_dir", type=str, default=join(config.CORPUS_DIR, "triviaqa"))
    parse.add_argument("-s", "--source", type=str, default=join(config.TRIVIA_QA, "evidence"))
    # This is slow, using more processes is recommended
    parse.add_argument("-n", "--n_processes", type=int, default=1)
    args = parse.parse_args()
    build_tokenized_corpus(args.source, NltkAndPunctTokenizer(), args.output_dir, n_processes=args.n_processes)

if __name__ == "__main__":
    main()