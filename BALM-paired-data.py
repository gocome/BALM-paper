import random


def shuffle_data(data_id, file_path):
    heavy_chains, light_chains = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            heavy_chain, light_chain = line.strip().split('</s>')
            heavy_chains.append(heavy_chain)
            light_chains.append(light_chain)

    random.shuffle(light_chains)

    file_path_shuffled = file_path.replace('.txt', '_shuffled.txt')
    lines = [f"{h}</s>{l}\n" for h, l in zip(heavy_chains, light_chains)]
    with open(file_path_shuffled, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    random.seed(42)

    # load the tran, eval, and test data
    data_files = {
        "train": ['./data/train-test-eval_paired/train.txt'],
        "eval": ['./data/train-test-eval_paired/eval.txt'],
        "test": ['./data/train-test-eval_paired/test.txt']
    }

    for data_id, file_paths in data_files.items():
        shuffle_data(data_id, file_paths[0])