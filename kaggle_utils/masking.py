class MaskingDataset(Dataset):
    def __init__(self, df, p_mask=0.2, categorical_cols=None):
        self.features = len(df.columns)
        self.p_mask = p_mask

        self._categorical_cols = []
        if categorical_cols is not None:
            for col in categorical_cols:
                one_hot = pd.get_dummies(df[col])
                one_hot[df[col].isnull()] = np.nan

                self._categorical_cols.append(one_hot.columns)
                df = df.drop(col, axis=1)
                df = pd.concat([df, one_hot], axis=1)

        self.df = df

    def __len__(self):
        return len(self.df)

    def onehot_categorical_mask(self, column_mask):
        return np.concatenate(
            [np.full(len(cs), m) for m, cs in zip(column_mask, self._categorical_cols)]
        )

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        output_mask = torch.tensor(~record.isnull().values)

        input_mask = ~np.random.binomial(1, self.p_mask, size=self.features).astype(
            bool
        )

        # mask continuous columns normally
        # mask categorical columns with all their columns

        full_input_mask = (
            torch.tensor(
                np.concatenate(
                    [
                        input_mask[: -len(self._categorical_cols)],
                        self.onehot_categorical_mask(
                            input_mask[-len(self._categorical_cols) :]
                        ),
                    ]
                )
            )
            & output_mask
        )

        x_in = record.values.copy()
        x_in[~full_input_mask] = 0

        x_out = record.values.copy()
        x_out[~output_mask] = np.nan

        # x = torch.tensor(record.values)
        # return masked_tensor(x, full_input_mask), masked_tensor(x, output_mask)
        return (
            torch.as_tensor(x_in, dtype=torch.float32),
            torch.as_tensor(x_out, dtype=torch.float32),
        )


def separate_slices(categorical_slices, output_size):
    slices = []
    slice_is_categorical = []
    prev_start = 0
    for start, end in categorical_slices:
        if start != prev_start:
            slices.append(slice(prev_start, start))
            slice_is_categorical.append(False)
        slices.append(slice(start, end))
        slice_is_categorical.append(True)
        prev_start = end

    if prev_start != output_size:
        slices.append(slice(prev_start, output_size))
        slice_is_categorical.append(False)

    return slices, slice_is_categorical


class MixedContCatModel(nn.Module):
    def __init__(self, model, categorical_slices, output_size) -> None:
        super().__init__()

        self.model = model
        
        self.softmax = nn.Softmax(dim=1)

        self.slices, self.slice_is_categorical = separate_slices(
            categorical_slices, output_size
        )

    def forward(self, x):
        x = self.model(x)
        for slice_, is_categorical in zip(self.slices, self.slice_is_categorical):
            if is_categorical:
                x[:, slice_] = self.softmax(x[:, slice_])
        return x