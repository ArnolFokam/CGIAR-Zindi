{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision.transforms import transforms\n",
    "from torch.utils.data import DataLoader\n",
    "from torch import optim\n",
    "\n",
    "# support local import\n",
    "import sys\n",
    "sys.path.append('../')\n",
    "\n",
    "from model import *\n",
    "from data import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs=100\n",
    "DATA_DIR=get_dir('../data')\n",
    "lr=0.0001\n",
    "output_dir=get_dir('.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: '/mnt/data/home/manuel/cgiar/data/train/L331F04163C39S12607Rp34788.jpg'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb Cell 3\u001b[0m line \u001b[0;36m2\n\u001b[1;32m     <a href='vscode-notebook-cell://ssh-remote%2Bmerman/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb#W2sdnNjb2RlLXJlbW90ZQ%3D%3D?line=21'>22</a>\u001b[0m \u001b[39mfor\u001b[39;00m epoch \u001b[39min\u001b[39;00m \u001b[39mrange\u001b[39m(epochs):\n\u001b[1;32m     <a href='vscode-notebook-cell://ssh-remote%2Bmerman/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb#W2sdnNjb2RlLXJlbW90ZQ%3D%3D?line=22'>23</a>\u001b[0m     model\u001b[39m.\u001b[39mtrain()\n\u001b[0;32m---> <a href='vscode-notebook-cell://ssh-remote%2Bmerman/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb#W2sdnNjb2RlLXJlbW90ZQ%3D%3D?line=23'>24</a>\u001b[0m     \u001b[39mfor\u001b[39;00m _, images, extents \u001b[39min\u001b[39;00m train_loader:\n\u001b[1;32m     <a href='vscode-notebook-cell://ssh-remote%2Bmerman/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb#W2sdnNjb2RlLXJlbW90ZQ%3D%3D?line=24'>25</a>\u001b[0m         optimizer\u001b[39m.\u001b[39mzero_grad()\n\u001b[1;32m     <a href='vscode-notebook-cell://ssh-remote%2Bmerman/mnt/data/home/manuel/cgiar/notebooks/v1.ipynb#W2sdnNjb2RlLXJlbW90ZQ%3D%3D?line=25'>26</a>\u001b[0m         outputs \u001b[39m=\u001b[39m model(images)\n",
      "File \u001b[0;32m~/anaconda3/envs/cgiar/lib/python3.8/site-packages/torch/utils/data/dataloader.py:633\u001b[0m, in \u001b[0;36m_BaseDataLoaderIter.__next__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    630\u001b[0m \u001b[39mif\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_sampler_iter \u001b[39mis\u001b[39;00m \u001b[39mNone\u001b[39;00m:\n\u001b[1;32m    631\u001b[0m     \u001b[39m# TODO(https://github.com/pytorch/pytorch/issues/76750)\u001b[39;00m\n\u001b[1;32m    632\u001b[0m     \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_reset()  \u001b[39m# type: ignore[call-arg]\u001b[39;00m\n\u001b[0;32m--> 633\u001b[0m data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49m_next_data()\n\u001b[1;32m    634\u001b[0m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_num_yielded \u001b[39m+\u001b[39m\u001b[39m=\u001b[39m \u001b[39m1\u001b[39m\n\u001b[1;32m    635\u001b[0m \u001b[39mif\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_dataset_kind \u001b[39m==\u001b[39m _DatasetKind\u001b[39m.\u001b[39mIterable \u001b[39mand\u001b[39;00m \\\n\u001b[1;32m    636\u001b[0m         \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_IterableDataset_len_called \u001b[39mis\u001b[39;00m \u001b[39mnot\u001b[39;00m \u001b[39mNone\u001b[39;00m \u001b[39mand\u001b[39;00m \\\n\u001b[1;32m    637\u001b[0m         \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_num_yielded \u001b[39m>\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_IterableDataset_len_called:\n",
      "File \u001b[0;32m~/anaconda3/envs/cgiar/lib/python3.8/site-packages/torch/utils/data/dataloader.py:677\u001b[0m, in \u001b[0;36m_SingleProcessDataLoaderIter._next_data\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    675\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39m_next_data\u001b[39m(\u001b[39mself\u001b[39m):\n\u001b[1;32m    676\u001b[0m     index \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_next_index()  \u001b[39m# may raise StopIteration\u001b[39;00m\n\u001b[0;32m--> 677\u001b[0m     data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49m_dataset_fetcher\u001b[39m.\u001b[39;49mfetch(index)  \u001b[39m# may raise StopIteration\u001b[39;00m\n\u001b[1;32m    678\u001b[0m     \u001b[39mif\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_pin_memory:\n\u001b[1;32m    679\u001b[0m         data \u001b[39m=\u001b[39m _utils\u001b[39m.\u001b[39mpin_memory\u001b[39m.\u001b[39mpin_memory(data, \u001b[39mself\u001b[39m\u001b[39m.\u001b[39m_pin_memory_device)\n",
      "File \u001b[0;32m~/anaconda3/envs/cgiar/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py:51\u001b[0m, in \u001b[0;36m_MapDatasetFetcher.fetch\u001b[0;34m(self, possibly_batched_index)\u001b[0m\n\u001b[1;32m     49\u001b[0m         data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdataset\u001b[39m.\u001b[39m__getitems__(possibly_batched_index)\n\u001b[1;32m     50\u001b[0m     \u001b[39melse\u001b[39;00m:\n\u001b[0;32m---> 51\u001b[0m         data \u001b[39m=\u001b[39m [\u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdataset[idx] \u001b[39mfor\u001b[39;00m idx \u001b[39min\u001b[39;00m possibly_batched_index]\n\u001b[1;32m     52\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[1;32m     53\u001b[0m     data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdataset[possibly_batched_index]\n",
      "File \u001b[0;32m~/anaconda3/envs/cgiar/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py:51\u001b[0m, in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     49\u001b[0m         data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdataset\u001b[39m.\u001b[39m__getitems__(possibly_batched_index)\n\u001b[1;32m     50\u001b[0m     \u001b[39melse\u001b[39;00m:\n\u001b[0;32m---> 51\u001b[0m         data \u001b[39m=\u001b[39m [\u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mdataset[idx] \u001b[39mfor\u001b[39;00m idx \u001b[39min\u001b[39;00m possibly_batched_index]\n\u001b[1;32m     52\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[1;32m     53\u001b[0m     data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdataset[possibly_batched_index]\n",
      "File \u001b[0;32m~/cgiar/notebooks/../data.py:37\u001b[0m, in \u001b[0;36mCGIARDataset.__getitem__\u001b[0;34m(self, idx)\u001b[0m\n\u001b[1;32m     36\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39m__getitem__\u001b[39m(\u001b[39mself\u001b[39m, idx):\n\u001b[0;32m---> 37\u001b[0m     image \u001b[39m=\u001b[39m Image\u001b[39m.\u001b[39;49mopen(\u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mimages_dir \u001b[39m/\u001b[39;49m \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mdf\u001b[39m.\u001b[39;49miloc[idx, \u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mcolumns\u001b[39m.\u001b[39;49mindex(\u001b[39m\"\u001b[39;49m\u001b[39mfilename\u001b[39;49m\u001b[39m\"\u001b[39;49m)])\u001b[39m.\u001b[39mconvert(\u001b[39m'\u001b[39m\u001b[39mRGB\u001b[39m\u001b[39m'\u001b[39m)\n\u001b[1;32m     39\u001b[0m     \u001b[39mif\u001b[39;00m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mtransform:\n\u001b[1;32m     40\u001b[0m         image \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mtransform(image)\n",
      "File \u001b[0;32m~/anaconda3/envs/cgiar/lib/python3.8/site-packages/PIL/Image.py:3218\u001b[0m, in \u001b[0;36mopen\u001b[0;34m(fp, mode, formats)\u001b[0m\n\u001b[1;32m   3215\u001b[0m     filename \u001b[39m=\u001b[39m fp\n\u001b[1;32m   3217\u001b[0m \u001b[39mif\u001b[39;00m filename:\n\u001b[0;32m-> 3218\u001b[0m     fp \u001b[39m=\u001b[39m builtins\u001b[39m.\u001b[39;49mopen(filename, \u001b[39m\"\u001b[39;49m\u001b[39mrb\u001b[39;49m\u001b[39m\"\u001b[39;49m)\n\u001b[1;32m   3219\u001b[0m     exclusive_fp \u001b[39m=\u001b[39m \u001b[39mTrue\u001b[39;00m\n\u001b[1;32m   3221\u001b[0m \u001b[39mtry\u001b[39;00m:\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/mnt/data/home/manuel/cgiar/data/train/L331F04163C39S12607Rp34788.jpg'"
     ]
    }
   ],
   "source": [
    "# Define transform for image preprocessing\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "# Create instances of CGIARDataset for training and testing\n",
    "train_dataset = CGIARDataset(root_dir=DATA_DIR, split='train', transform=transform)\n",
    "\n",
    "# Create DataLoader instances\n",
    "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "\n",
    "# Initialize the regression model\n",
    "model = Resnet50_V1()\n",
    "\n",
    "# Define loss function (mean squared error) and optimizer (e.g., Adam)\n",
    "criterion = nn.MSELoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=lr)\n",
    "\n",
    "# Training loop (you can customize this loop)\n",
    "for epoch in range(epochs):\n",
    "    model.train()\n",
    "    for _, images, extents in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs.squeeze(), extents.float())\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.eval()\n",
    "\n",
    "test_dataset = CGIARDataset(root_dir='data', split='test', transform=transform)\n",
    "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)\n",
    "predictions = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for ids, images, _ in test_loader:\n",
    "        outputs = model(images)\n",
    "        outputs = outputs.squeeze().tolist()\n",
    "        predictions.extend(list(zip(ids, outputs)))\n",
    "\n",
    "# create a dictionar from the pairs of id and prediction\n",
    "predictions = dict(predictions)\n",
    "\n",
    "# load the sample submission file and update the extent column with the predictions\n",
    "submission_df = pd.read_csv('data/SampleSubmission.csv')\n",
    "\n",
    "# update the extent column with the predictions\n",
    "submission_df['extent'] = submission_df['id'].map(predictions)\n",
    "\n",
    "# save the submission file and trained model\n",
    "submission_df.to_csv(output_dir / 'submission.csv', index=False)\n",
    "torch.save(model.state_dict(), output_dir / 'model.pth')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "cgiar",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
