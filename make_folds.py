from tqdm import tqdm

new_csv = {}
new_csv['ImageId'] = []
new_csv[' EncodedPixels'] = []


PATH = "../../data/ibespalov/SIIM_ACR/train-rle.csv"
PATH_TO_SAVE = "../../data/ibespalov/SIIM_ACR/train.csv"

mask = pd.read_csv(PATH)

k = mask.groupby('ImageId').count()[" EncodedPixels"]
list_ImageId = k.index[k>1]

for i in tqdm(range(len(list_ImageId))):
    tmp = mask.loc[mask['ImageId'] == list_ImageId[i]]
    final_mask = np.zeros((1024, 1024))
    for index, row in tmp.iterrows():
        RLE_mask = row[" EncodedPixels"]
        if RLE_mask.strip() != str(-1):
            rle_mask = rle2mask(RLE_mask[1:], 1024, 1024).T
        else:
            rle_mask = np.zeros((1024, 1024))
        final_mask[np.where(rle_mask)] = 1
    final_rle = mask_to_rle(final_mask.T, 1024, 1024)

    new_csv['ImageId'].append(list_ImageId[i])
    new_csv[' EncodedPixels'].append(final_rle)

k = mask.groupby('ImageId').count()[" EncodedPixels"]
list_ImageId = k.index[k==1]

for i in tqdm(range(len(list_ImageId))):
    new_csv['ImageId'].append(list_ImageId[i])
    new_csv[' EncodedPixels'].append(mask.loc[mask['ImageId'] == list_ImageId[i]][" EncodedPixels"].values[0])

new_df = pd.DataFrame(data=new_csv)
df = new_df
df = df.sample(frac=1).reset_index(drop=True)
df['fold'] = [ i % 10 for i in np.arange(len(df))]
df.to_csv(PATH_TO_SAVE, index=False)
