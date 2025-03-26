from urllib.error import HTTPError

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


# a Dataset class must implement the following 3 methods
class GPTDatasetV1(Dataset):
    def __init__(
        self,
        txt: str,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
        annotate=None,
    ):
        super().__init__()
        self.input_ids = []  # token ids
        self.target_ids = []  # token ids

        encoded_txt = tokenizer.encode(txt)
        token_ids = encoded_txt.ids
        print(
            f"Number of tokens {'in ' + annotate if annotate else ''}: {len(token_ids)}"
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[
                i + 1 : i + 1 + max_length
            ]  # predict the next set of words
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    dataset: Dataset, batch_size: int, shuffle=True, drop_last=True, num_workers=0
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader


if __name__ == "__main__":
    txt = "Hello, what's up GENAI?"

    tokenizer_model = "something"
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_model)

    except HTTPError as err:
        raise err

    else:
        dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=4, stride=1)

        dataloader = create_dataloader_v1(
            dataset=dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0
        )
