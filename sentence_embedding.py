from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import pandas as pd
import re


def read_excel_file(file_path, sheet_name=0):
    if file_path:
        df = pd.read_excel(io=file_path, sheet_name=sheet_name, usecols=[0], names=None)
        results = []
        for s_li in df.values.tolist():
            results.append(s_li[0])
        return results
    else:
        print('未输入文件地址')


def get_comments(item_str):
    item_lines = item_str.split("\n")
    print(f"before process len:{len(item_lines)}")
    comments = []
    comment = item_lines[0]
    pattern = re.compile(r".*\(.*\):")

    for i, line in enumerate(item_lines[1:]):
        M = pattern.match(line)
        if M is not None:
            comments.append(comment.split(":")[1].strip())
            comment = line
        else:
            comment += line
    comments.append(comment.split(":")[1].strip())
    print(f"after process len:{len(comments)}")
    return comments


def main():
    df = read_excel_file("sen.xlsx")
    print(f"len df:{len(df)}")
    null_count = 0
    sen_list = []
    sen_list_indices = []
    for i, d in enumerate(df):
        if d != "(null)" and d != "(空字符串)":
            comments = get_comments(d)
            sen_list.append(comments)
            sen_list_indices.append(i)
        else:
            null_count += 1
    print(f"len sen_list:{len(sen_list)}, null comment:{null_count}")

    # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    # These models find semantically similar sentences within one language or across languages:
    #
    # distiluse-base-multilingual-cased-v1: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. Supports 15 languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish.
    #
    # distiluse-base-multilingual-cased-v2: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. This version supports 50+ languages, but performs a bit weaker than the v1 model.
    #
    # paraphrase-multilingual-MiniLM-L12-v2 - Multilingual version of paraphrase-MiniLM-L12-v2, trained on parallel data for 50+ languages.
    #
    # paraphrase-multilingual-mpnet-base-v2 - Multilingual version of paraphrase-mpnet-base-v2, trained on parallel data for 50+ languages.

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Max Sequence Length:", model.max_seq_length)

    # Embed a list of sentences
    output = []
    for i, comments in enumerate(sen_list):
        assert type(comments) == list
        assert len(comments) >= 1
        sentences = comments

        sentence_embeddings = model.encode(sentences)
        print(type(sentence_embeddings))
        print(sentence_embeddings.shape)

        comment_embedding = np.mean(sentence_embeddings, axis=0)
        assert comment_embedding.shape == (768, )

        output.append(comment_embedding)
        # The result is a list of sentence embeddings as numpy arrays
        # for sentence, embedding in zip(sentences, sentence_embeddings):
        #     print("Sentence:", sentence)
        #     print("Embedding:", embedding)
        #     print(type(embedding))
    print(f"len embedding:{len(output)}")
    assert len(sen_list) == len(output)
    np.savetxt("comments_vectors_single_comment.txt", output)
    np.savetxt("comments_indices_single_comment.txt", sen_list_indices)

if __name__ == "__main__":
    main()