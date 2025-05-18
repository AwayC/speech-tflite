import numpy as np
import os

def convert_txt_to_npy(input_path, output_dir):
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    groups = []
    i = 0
    while i < len(lines):
        if lines[i] == 'st':
            # 尝试获取接下来的80行数据
            group_lines = []
            for j in range(i+1, i+81):  # 严格取80行
                if j >= len(lines) or lines[j] == 'st':
                    break
                group_lines.append([float(x) for x in lines[j].split()])
            
            # 验证是否为有效的80x12数据
            if len(group_lines) == 80 and all(len(row)==12 for row in group_lines):
                groups.append(np.array(group_lines))
            else:
                print(f"跳过不完整组，起始行号 {i}，实际获取行数 {len(group_lines)}")
            
            i = j  # 跳到下一个组的起始位置
        else:
            i += 1
    
    # 保存有效组
    os.makedirs(output_dir, exist_ok=True)
    for idx, group in enumerate(groups):
        np.save(os.path.join(output_dir, f'group_{idx}.npy'), group)
        print(f"已保存 {output_dir}/group_{idx}.npy 形状: {group.shape}")

# 使用示例
convert_txt_to_npy("./data-raw/class0/data.txt", "./speech-classes/class0")
convert_txt_to_npy("./data-raw/class1/data.txt", "./speech-classes/class1")
convert_txt_to_npy("./data-raw/class2/data.txt", "./speech-classes/class2")
convert_txt_to_npy("./data-raw/class3/data.txt", "./speech-classes/class3")
convert_txt_to_npy("./data-raw/class4/data.txt", "./speech-classes/class4")
