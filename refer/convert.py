import pandas as pd

def convert_triplet_to_pair(input_csv, output_csv):
    # 定义类型映射
    imglabel = {"fd": 1, "fs": 2, "md": 3, "ms": 4, "nonkin": 0, "bb": 5, "ss": 6, "sibs": 7}

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 创建存储转换后的数据的列表
    pairs = []

    # 遍历每一行，拆分三元组为两组pair
    for _, row in df.iterrows():
        anchor = row['Anchor']
        positive = row['Positive']
        negative = row['Negative']
        kinship_type = row['ptype']

        # 添加Anchor-Positive pair
        pairs.append({'anchor': anchor, 'person': positive, 'type': imglabel.get(kinship_type, -1)})

        # 添加Anchor-Negative pair，type为nonkin
        pairs.append({'anchor': anchor, 'person': negative, 'type': imglabel['nonkin']})

    # 转换为DataFrame
    pairs_df = pd.DataFrame(pairs)

    # 保存到新的CSV文件
    pairs_df.to_csv(output_csv, index=False)
    print(f"Conversion complete. Saved to {output_csv}")
    


input_csv = "data\hand_cleaned_filtered_triplets_with_labels2_corrected.csv"  # 输入文件路径
output_csv = "data\pair.csv"    # 输出文件路径
convert_triplet_to_pair(input_csv, output_csv)
