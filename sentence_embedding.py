from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re


def read_excel_file(file_path, sheet_name=0):
    """
    读取excel第一列数据到数组中，不带表头
    :param file_path: excel文件地址
    :param sheet_name: 读取的sheet名，默认是第一张表
    :return:
    """
    if file_path:
        df = pd.read_excel(io=file_path, sheet_name=sheet_name, usecols=[0], names=None)
        results = []
        for s_li in df.values.tolist():
            results.append(s_li[0])
        return results
    else:
        print('未输入文件地址')


def get_single_role_comments(row_str):
    """
    获取 Legal 角色评论内容
    :return:
    """
    comments = []

    pattern = re.compile(r".*\(.*\):")
    line_list = row_str.split("\n")
    comment = line_list[0]

    for i, line in enumerate(line_list[1:]):
        match_res = pattern.match(line)
        if match_res is not None:
            comments.append(comment)  # 整句话加进去，包括角色
            comment = line
        else:
            comment += line
    comments.append(comment)
    for i, comment in enumerate(comments):
        c_list = comment.split(":")
        if "法务" in c_list[0]:  # 角色中包含法务
            return [c_list[1].strip()]
    return []  # 说明没有法务角色在评论,返回空


def get_full_comments(row_str):
    """
    获取所有角色评论内容
    :param row_str:
    :return:
    """
    comments = []

    pattern = re.compile(r".*\(.*\):")
    item_lines = row_str.split("\n")
    comment = item_lines[0]
    for i, line in enumerate(item_lines[1:]):
        match_res = pattern.match(line)
        if match_res is not None:  # 说明是新的角色在评论
            comments.append(comment.split(":")[1].strip())
            comment = line
        else:
            comment += line
    comments.append(comment.split(":")[1].strip())
    return comments


def main():
    # step1: 数据加载
    rows = read_excel_file("sen.xlsx")
    print(f"--------清洗前数据条数:{len(rows)}--------")

    # step2: 数据清洗
    null_count = 0
    no_legal_cnt = 0
    sen_list = []  # 存储清洗后的sentence
    sen_index_list = []  # 存储清洗后的sentence在原表格中的行号
    for i, row in enumerate(rows):
        if row != "(null)" and row != "(空字符串)":
            # comments = get_full_comments(row)
            comments = get_single_role_comments(row)
            if len(comments) >= 1:
                sen_list.append(comments)
                sen_index_list.append(i)
            else:
                no_legal_cnt += 1
        else:
            null_count += 1
    print(f"--------清洗后数据条数:{len(sen_list)}, 空数据条数:{null_count}, 无legal评论数据条数:{no_legal_cnt}--------")

    # step3: 提取特征向量
    embedding_res = []
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    for i, sen in enumerate(sen_list):
        assert type(sen) == list
        assert len(sen) >= 1

        sentence_embeddings = model.encode(sen)
        # print(type(sentence_embeddings), "\n", sentence_embeddings.shape)
        comment_embedding = np.mean(sentence_embeddings, axis=0)
        assert comment_embedding.shape == (768,)
        embedding_res.append(comment_embedding)

    print(f"--------embedding数据条数:{len(embedding_res)}--------")
    assert len(sen_list) == len(embedding_res)

    # step4: 数据保存
    # np.savetxt("comments_vectors.txt", embedding_res)
    # np.savetxt("comments_indices.txt", sen_index_list)
    np.savetxt("comments_vectors_single_comment.txt", embedding_res)
    np.savetxt("comments_indices_single_comment.txt", sen_index_list, fmt='%i')


if __name__ == "__main__":
    main()
