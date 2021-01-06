import os
import cv2
import shutil
import random
import argparse
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.unet import SCSEUnet

gpu_ids = '0, 1, 2, 3, 4, 5'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

np.random.seed(666666)
patch_size = '512'  # The input size of training data
train_fold_best_val_record = np.load('train_fold_best_val_record.npy', allow_pickle=True)

val_fold_1 = ['053930d5-78de-4441-bd39-aa731ca984fa.jpg', '088df053-db35-4f2c-a94f-ab12b442d489.jpg',
         '09bb5e09-f755-464b-b94f-d6a3f6831895.jpg', '0be69a15-08b1-4259-aaed-e55f8bcd019f.jpg',
         '0d620c85-80c1-46fe-8f34-d6047977c27e.jpg', '110.jpg', '113.jpg', '115.jpg', '118.jpg', '12.jpg', '132.jpg',
         '135.jpg', '137.jpg', '139.jpg', '14636725-4fd6-41f3-b436-d540af68fd92.jpg', '151.jpg',
         '156b4044-a3ae-4214-8b53-6aed556e5f0d.jpg', '158.jpg', '164.jpg', '169.jpg',
         '16d1fde5-4254-495b-a15d-1e981e7e8d8f.jpg', '179.jpg', '189.jpg', '192.jpg', '194.jpg',
         '1b10db07-9b6e-41d3-8cdf-86c089109d97.jpg', '203.jpg', '229.jpg', '236.jpg', '245.jpg', '249.jpg',
         '24c4fa77-554d-4e52-8f1f-be31e4fa534a.jpg', '251.jpg', '282.jpg', '285.jpg', '291.jpg', '293.jpg',
         '29b819b6-3454-4828-b424-7a05fccb0a37.jpg', '2c59b12e-d780-42e9-8a3f-302973cf9247.jpg',
         '2f43a901-115b-4700-80f5-4233a1e4f66d.jpg', '310.jpg', '32762b56-5de7-4cd0-b59d-bffd9cf5bfbb.jpg', '331.jpg',
         '333.jpg', '348.jpg', '354.jpg', '366.jpg', '369.jpg', '3eca1380-d271-48a1-9784-b644a28da4fd.jpg', '404.jpg',
         '418.jpg', '42.jpg', '422.jpg', '428.jpg', '429.jpg', '43.jpg', '456.jpg', '468.jpg',
         '46e034bb-9eb3-4160-9b83-1c94a2ffcf92.jpg', '46f38886-8ad4-44b9-ba35-b78b56d617bc.jpg', '479.jpg', '487.jpg',
         '493.jpg', '4c11b54a-23a5-44e9-8def-fcbacc06c8b3.jpg', '518.jpg', '528.jpg', '538.jpg', '546.jpg', '556.jpg',
         '56.jpg', '561.jpg', '562.jpg', '570.jpg', '573.jpg', '574.jpg', '577.jpg',
         '5859e3f2-b383-40ac-aeb2-23879a794ea4.jpg', '587.jpg', '59.jpg', '595.jpg', '599.jpg', '60.jpg', '602.jpg',
         '603.jpg', '607.jpg', '65.jpg', '665.jpg', '667.jpg', '674.jpg', '682.jpg', '687.jpg',
         '68a73938-d68a-4a6b-943b-fd0f3bce36a2.jpg', '70e23cde-d6c6-4404-86cd-de4766da4305.jpg', '710.jpg', '723.jpg',
         '733.jpg', '73bb0159-7286-40fd-9e19-cca0cd460f00.jpg', '741.jpg', '751.jpg', '760.jpg', '763.jpg', '78.jpg',
         '781.jpg', '786.jpg', '78812033-bfa1-4e87-998b-7e1ad9dfe726.jpg', '794.jpg',
         '7b9cf209-8734-4357-a484-a211f1e09f4a.jpg', '7bc6e3d2-db16-4934-ae77-8dbbac12839b.jpg', '80.jpg', '800.jpg',
         '8075d956-dca1-40f7-b425-e7eb88629eda.jpg', '807c1533-15a3-4362-8a9f-a86b5b9c4797.jpg', '817.jpg', '820.jpg',
         '828.jpg', '82cecc15-acb8-4561-b817-762af3745b31.jpg', '840.jpg', '847.jpg', '858.jpg', '880.jpg', '885.jpg',
         '889.jpg', '88f09d9d-1b57-44a0-b263-9b65d4ca2e1a.jpg', '8941743f-6d6a-4880-b7a3-a45cf7248d39.jpg', '896.jpg',
         '8e4114ed-f6ca-4133-95fa-f7f56184a145.jpg', '903.jpg', '917.jpg', '944.jpg', '947.jpg',
         '9480f983-9bea-46da-bea2-33fab9bbb436.jpg', '95.jpg', '959.jpg', '964.jpg', '971.jpg',
         '9727f6b9-de25-4749-a352-db60080a44bb.jpg', '974.jpg', '979.jpg', '981.jpg',
         '99b6c7ed-b9f8-412f-a219-1be33b101bf7.jpg', '9f9adc91-d25d-4a4b-9096-bcad8f6c2e48.jpg',
         '9fd08b67-a18b-4afe-ba4e-a29132c29ea2.jpg', 'a22a918a-2ad7-499e-8099-3f4ca1017538.jpg',
         'a3728ebb-1d73-41d6-a1ba-6f82a6497589.jpg', 'ae1a7f3d-9f04-4d73-89be-4dc85c6d1d1e.jpg',
         'b1eec58a-0667-4927-bb7c-67c7bae98f25.jpg', 'b297198f-1638-4a6d-8389-f7d4871fa94f.jpg',
         'bdcab4fe-0e9d-4e37-81f2-79a8c0f5bca9.jpg', 'c3186a7a-246e-4268-81d7-c0717104455c.jpg',
         'd2b89c28-cb5f-4d3b-8250-fcc6d1b39172.jpg', 'd2c31c90-50a9-4b96-9207-db560a3f7e4b.jpg',
         'd8184388-24b4-467f-8aa0-d6e2233c3bab.jpg', 'd9aed78f-769e-4081-be8d-93ead4b6b781.jpg',
         'de282868-bb6b-47c2-a9c5-23d667805689.jpg', 'e0409331-4575-49b0-8caf-4c5ad79312b0.jpg',
         'e846272e-f17a-48b9-aae6-2de1d6e2ff53.jpg', 'e8d379c1-e813-446e-9c44-04a2d00fde54.jpg',
         'ec240ada-432e-4698-a087-bbf865334bb4.jpg', 'fba9da57-e39d-4234-bd62-b90a5089f7a4.jpg',
         'fdf1f2cd-fa0e-452b-9df1-17844daec437.jpg', 'fe21422d-d6ab-4089-8c05-4ca091a70e85.jpg',
         'fe99a8d1-fbe3-4c7b-8439-42f2d4b30e0f.jpg']
val_fold_2 = ['0490b9bf-d25b-4c21-9b87-580a2d38e6f4.jpg', '08b6c0d9-d05f-4a10-8cdc-89ef8cbca80d.jpg',
         '09cb6966-387e-4067-a7c8-07b3fcaa645c.jpg', '0cb94749-bebd-4b63-a649-096b1cbe5e20.jpg', '10.jpg', '1000.jpg',
         '102e445c-523b-424b-b6f7-b54da4c51e23.jpg', '104.jpg', '111.jpg', '11fce814-4db5-4cab-8684-8bc2649dd015.jpg',
         '123.jpg', '128.jpg', '136.jpg', '14.jpg', '140.jpg', '142.jpg', '15.jpg',
         '15221a21-88ae-467e-9f4b-7e09fd1ec8b1.jpg', '165.jpg', '1696c4b0-6688-4a5f-9ae1-79f59e819d83.jpg', '172.jpg',
         '174.jpg', '175.jpg', '176.jpg', '18583047-7822-4e56-b458-84131ca79955.jpg', '19.jpg', '193.jpg',
         '19655862-f943-4e09-9154-def0843825bb.jpg', '197.jpg', '1d7572cd-ca34-46e9-8151-84c7a5316a69.jpg', '2.jpg',
         '200.jpg', '22.jpg', '227.jpg', '234.jpg', '234a2229-7218-4a4f-bb95-35c7bc0260ee.jpg', '240.jpg', '25.jpg',
         '253.jpg', '258.jpg', '259.jpg', '265.jpg', '268.jpg', '273.jpg', '292.jpg', '299.jpg', '3.jpg', '30.jpg',
         '300.jpg', '305.jpg', '31.jpg', '312.jpg', '318.jpg', '322.jpg', '328.jpg',
         '32b80422-b165-46a8-a80d-a8d101cc5a91.jpg', '34.jpg', '341daf20-6a95-4a02-8bfd-839e3b67b588.jpg', '356.jpg',
         '358.jpg', '362.jpg', '363.jpg', '36e5eddf-022b-47e7-b9d6-8151d98a14c0.jpg', '37.jpg', '378.jpg', '379.jpg',
         '37e034aa-91f3-4dcf-a954-e545093cfb40.jpg', '382.jpg', '383.jpg', '38c844e2-0972-457c-a387-230e64e82cff.jpg',
         '396.jpg', '397.jpg', '398.jpg', '3b014a9f-fa89-4c7f-9b66-e2be9fe9cc22.jpg',
         '3b75c691-79d8-4709-9439-44da222a1044.jpg', '3e3d5af8-8427-4d1c-bc87-6976178f1579.jpg',
         '400deafd-2202-4013-a537-346615da6692.jpg', '41.jpg', '412.jpg', '420.jpg', '425.jpg', '427.jpg', '435.jpg',
         '43b26e4e-8866-4316-9a37-1e019465b00d.jpg', '440.jpg', '447.jpg', '450.jpg', '460.jpg', '466.jpg',
         '46f0c9cb-7618-46d2-a05c-c390ed40b97a.jpg', '475.jpg', '478.jpg', '483.jpg', '486.jpg', '492.jpg', '496.jpg',
         '4aaf81d1-6a1c-4980-87bc-f48fd8fd0809.jpg', '4b85cf1c-6114-4b37-8c5a-5d8cf5290f19.jpg', '509.jpg',
         '50936255-59da-4380-8150-22b8a4c1c008.jpg', '515.jpg', '516.jpg', '52.jpg', '522.jpg', '525.jpg',
         '527fac09-79aa-41af-9359-af3fde56b05e.jpg', '535.jpg', '539.jpg', '54.jpg', '544.jpg', '554.jpg', '560.jpg',
         '566.jpg', '567.jpg', '580.jpg', '589.jpg', '591.jpg', '594.jpg', '597.jpg',
         '5c3571e6-1893-4e3d-8aba-19035a16c589.jpg', '5cf2e4c8-7ce2-4e88-908a-097bcf3ffa4c.jpg',
         '5ede9a6e-8397-412c-a2dc-5d405d18fabb.jpg', '6.jpg', '605c4d75-f62e-4a6a-a269-9d0946e5c993.jpg',
         '60a6af4c-d0ab-4920-9231-ab0fd4e44ae9.jpg', '6146acf3-efd6-48e5-91c8-7d4c2fb46bd7.jpg', '616.jpg', '620.jpg',
         '6288a22e-ca0c-4c1e-a6a7-de54cce5202f.jpg', '635.jpg', '64.jpg', '642bf520-a108-498d-af70-8b6bc0577b0d.jpg',
         '648.jpg', '653.jpg', '661.jpg', '663.jpg', '668.jpg', '669.jpg', '671.jpg', '681.jpg', '686.jpg', '688.jpg',
         '690.jpg', '693.jpg', '695.jpg', '696.jpg', '700.jpg', '71.jpg', '711.jpg', '713.jpg', '714.jpg', '726.jpg',
         '72a8d5d1-2a13-4c45-b0b6-ec9eb5723900.jpg', '731.jpg', '734.jpg', '7368f1b5-d66e-488b-9fb9-5cbd148b91ea.jpg',
         '739.jpg', '754.jpg', '758.jpg', '761.jpg', '766a1372-d434-49a5-8058-0d22b9ee3426.jpg',
         '7681dbf7-108e-4433-936e-97cf8ac85562.jpg', '771.jpg', '778.jpg', '783.jpg', '788.jpg',
         '78c57261-b7bc-439d-9100-c9eeef7e2142.jpg', '792.jpg', '7b51218a-c8d0-4daa-b8ef-61e5cab0911d.jpg',
         '7fcd39f3-03ee-429d-9054-40d61da47510.jpg', '801.jpg', '804b31ec-0c8b-48e6-bc33-342089bf2d18.jpg', '805.jpg',
         '80823d9f-959a-4434-8972-c7f2e216fa49.jpg', '811.jpg', '813.jpg', '814.jpg', '82.jpg', '821.jpg',
         '82a90475-79a3-426b-9f2b-bc0cee0bcfd8.jpg', '832.jpg', '836.jpg', '840ff556-337d-45df-ab7b-28e0e7ec74a4.jpg',
         '842.jpg', '844.jpg', '846.jpg', '85.jpg', '852.jpg', '852cfce6-c0bc-4234-80f4-115f14057ab1.jpg', '859.jpg',
         '864.jpg', '868.jpg', '86c0eab0-ea07-4e8d-81ac-93e481173288.jpg', '871.jpg', '888.jpg', '899.jpg',
         '8b0fdd33-a3df-4bb7-8f26-d6f44377645e.jpg', '8c22f3c6-8492-4b8b-9d47-63649e3bbb64.jpg',
         '8fe130e5-df54-499f-839a-3980ffa9035b.jpg', '901879af-9118-4548-8cfd-dcba165edfa8.jpg', '904.jpg', '91.jpg',
         '91123aaf-e460-47c9-b0f5-c6a77b228768.jpg', '912.jpg', '914.jpg', '92.jpg', '938.jpg', '939.jpg',
         '93d10fef-416f-46a1-96ea-92d6303e0e9d.jpg', '941.jpg', '94a793a8-c5fd-4945-a285-59da45848ee5.jpg', '952.jpg',
         '953.jpg', '961.jpg', '963.jpg', '969.jpg', '97.jpg', '975.jpg', '975597bf-ad27-4192-8cdf-ff7e55706df4.jpg',
         '978.jpg', '984.jpg', '991adc7e-a3a3-4635-aba5-24974d79e3dd.jpg', '99a3b333-3400-40e2-80e1-dd78988cf665.jpg',
         '9a114854-be37-4481-9926-7c2223db9673.jpg', '9a258d8f-ed33-4e89-b76d-37bdd0b71385.jpg',
         '9c5039a9-89e2-44c4-a6fa-b56d3e819336.jpg', 'a03141c2-6126-41ad-b21d-dfd22aaccf80.jpg',
         'a314955a-d8be-48e3-a799-568278859a33.jpg', 'a5f60e10-a64e-4697-a13e-3454bf23db2d.jpg',
         'a776349b-95f8-4201-9c76-532c89e5604c.jpg', 'aa328860-8594-4966-b59c-4ded38cb52a7.jpg',
         'aa66e27f-f50d-4f3d-bf40-22b828fa36d6.jpg', 'aa6dadf2-72e8-4494-bf86-5b575b166601.jpg',
         'aa8e600f-b621-4372-b552-4950fb7d7ddc.jpg', 'ab32400c-dcf0-40aa-a362-0796c477d5de.jpg',
         'ac6fbfa3-136e-4b3e-9036-f7f59d8c7995.jpg', 'acd88be5-dca1-42a5-932d-eb891da71304.jpg',
         'ad20cc53-11c2-41bb-ba1c-1c73280b4d07.jpg', 'af95a7b1-5af1-457b-b8ea-11124a962361.jpg',
         'b195f794-d032-4174-91fb-fb57322e22a1.jpg', 'b6e97b95-6b2a-418b-a06c-3db4e3b10f42.jpg',
         'b6eb4c62-bb06-4555-a557-bb2d2ffca7eb.jpg', 'b99b35dd-a8de-4d2d-a8e1-7deda8bf19f1.jpg',
         'bb80d22f-6fea-4acd-bfcb-f4678b959649.jpg', 'bb821066-88d4-4e2e-ab05-cdb44b32b8f6.jpg',
         'c0f797cb-ceb4-4a1f-a789-4bb28b25395d.jpg', 'c131728f-95a0-4a6f-9f06-efef767fa350.jpg',
         'c1738ba2-4f9d-4e4c-af8d-495b9d1ec058.jpg', 'c29ee8d6-6783-4966-b650-59d9bfecfd91.jpg',
         'c51e325f-3b06-4a21-bd2c-094a80ddb38d.jpg', 'c76b0609-5533-48c7-8562-d996b5e3f520.jpg',
         'cb05ad0d-e223-431e-958f-f8f66c6ed532.jpg', 'cb435a17-3e93-4381-857a-f58ee58106b8.jpg',
         'cd83e414-dd04-4f1c-9a2b-144ddfa39c3f.jpg', 'ce3c75b3-772a-42ab-bb8e-860049518dd8.jpg',
         'cfb7b5fa-1f80-4bfa-9f17-4ddde6cad45f.jpg', 'd048ce24-e549-46fe-95ff-213842289314.jpg',
         'd1aaf535-0f9c-45b5-8805-d46938a3b09e.jpg', 'd27b81e0-9ca4-4989-8360-81c50484fef5.jpg',
         'd36d09cd-686a-49d2-88c0-844a8458ff54.jpg', 'd5e11c04-8b05-4cb7-9d71-b72626f89a03.jpg',
         'd7d12aa4-1e25-4cb3-8bdc-4136fb051688.jpg', 'da595c2e-8fe1-4755-90f7-f68f60b2e15b.jpg',
         'dc275e1c-250f-4f91-ad4c-bb31444b2947.jpg', 'deb42e3a-06dd-4fce-afeb-97c980a35787.jpg',
         'e12fb189-481f-40dd-9b83-92b0e259c1da.jpg', 'e232ebd9-726c-4d74-b3cc-53a5b7da713d.jpg',
         'e47b1c06-1e74-4bf9-989c-b42353010836.jpg', 'e5236e06-0999-4e73-a2ff-e9ba0ef78900.jpg',
         'e54e209a-6ac1-458b-bc40-73e72fe75cc8.jpg', 'e9cb7ea2-2835-441b-8c43-0b6d2abd1610.jpg',
         'ec172f4e-ba4e-405b-b9e5-34f077b76192.jpg', 'edb1f95c-8b07-4a30-9550-a7eff57cf297.jpg',
         'edbf3647-f9b2-41f4-85c1-f2506c5224b4.jpg', 'ef38245e-a385-48d4-b069-9a02ba8b28d8.jpg',
         'f011ce2f-de92-4a9c-b7bd-51c78e93a904.jpg', 'f8afed78-9e84-4387-a818-26558bcae619.jpg',
         'fc7bfd6b-f90a-4c53-a1ed-7faee0676376.jpg']


class GIID_Dataset(Dataset):
    def __init__(self, num=0, file='', choice='train', test_path='', tta_idx=1):
        self.num = num
        self.choice = choice
        if self.choice == 'test':
            self.test_path = test_path
            self.filelist = sorted(os.listdir(self.test_path))
        else:
            self.filelist = file

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.tta_idx = tta_idx

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        if self.choice == 'test':
            return len(self.filelist)
        return self.num

    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.test_path + self.filelist[idx], ''

        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)

        if self.tta_idx == 5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if self.tta_idx == 2 or self.tta_idx == 6:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif self.tta_idx == 3 or self.tta_idx == 7:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.tta_idx == 4 or self.tta_idx == 8:
            img = cv2.rotate(img, cv2.cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_180)

        if self.choice == 'train':
            img, mask = self.aug(img, mask)

        H, W, _ = img.shape
        size = int(patch_size)
        if self.choice == 'train' and (H != int(patch_size) or W != int(patch_size)):
            x = 0 if H == int(patch_size) else np.random.randint(0, H-size)
            y = 0 if W == int(patch_size) else np.random.randint(0, W-size)
            img = img[x:x + size, y:y + size, :]
            mask = mask[x:x + size, y:y + size, :]
        elif self.choice == 'val' and (H != 256 or W != 256):
            img = img[(H-size)//2:(H-size)//2+size, (W-size)//2:(W-size)//2+size, :]
            mask = mask[(H-size)//2:(H-size)//2+size, (W-size)//2:(W-size)//2+size, :]

        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    def aug(self, img, mask):
        # Resize the training data if necessary
        H, W, _ = img.shape
        if H < int(patch_size) or W < int(patch_size):
            m = int(patch_size) / min(H, W)
            img = cv2.resize(img, (int(H * m) + 1, int(W * m) + 1))
            mask = cv2.resize(mask, (int(H * m) + 1, int(W * m) + 1))
            mask[mask < 127.5] = 0
            mask[mask >= 127.5] = 255

        # Resize
        if random.random() < 0.5:
            H, W, C = img.shape
            if H * 0.9 > int(patch_size) and W * 0.9 > int(patch_size):
                r1, r2 = np.random.randint(90, 110) / 100., np.random.randint(90, 110) / 100.
            else:
                r1, r2 = np.random.randint(101, 110) / 100., np.random.randint(101, 110) / 100.
            img = cv2.resize(img, (int(H * r1), int(W * r2)))
            mask = cv2.resize(mask, (int(H * r1), int(W * r2)))
            mask[mask < 127.5] = 0
            mask[mask >= 127.5] = 255

        # Flip
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            tmp = random.random()
            if tmp < 0.33:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            elif tmp < 0.66:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.rotate(img, cv2.cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_180)

        # Noise
        if random.random() < 0.2:
            H, W, C = img.shape
            Nd = np.random.randint(10, 50) / 1000.
            # Nd = np.random.randint(30, 88) / 1000.
            Sd = 1 - Nd
            m = np.random.choice((0, 1, 2), size=(H, W, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            m = np.repeat(m, C, axis=2)
            m[mask == 0] = 2
            img[m == 0] = 0
            img[m == 1] = 255
        if random.random() < 0.2:
            H, W, C = img.shape
            N = np.random.randint(10, 50) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
            # N = np.random.randint(30, 88) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
            N = np.repeat(N, C, axis=2)
            img = img.astype(np.int32)
            img[mask == 255] = N[mask == 255] + img[mask == 255]
            img = N + img
            img[img > 255] = 255
            img = img.astype(np.uint8)

        return img, mask

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class GIID_Model(nn.Module):
    def __init__(self):
        super(GIID_Model, self).__init__()
        self.lr = 1e-4
        self.networks = SCSEUnet(backbone_arch='senet154')
        self.gen = nn.DataParallel(self.networks).cuda()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.save_dir = '../user_data/model_data/'
        self.bce_loss = nn.BCELoss()

    def process(self, Ii, Mg):
        self.gen_optimizer.zero_grad()

        Mo = self(Ii)

        gen_loss = self.bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
        return Mo, gen_loss

    def forward(self, Ii):
        return self.gen(Ii)

    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'GIID_weights.pth')

    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'GIID_weights.pth'))


class ForgeryForensics():
    def __init__(self, fold=1):
        self.fold = fold
        self.test_num = 1
        self.batch_size = 18

        if fold == 1:
            # Train num: 2328 + 491 = 2819
            self.train_npy = 'train_1o5_41x10_72x10.npy'
            self.train_file = np.load('../user_data/flist/' + self.train_npy)
            self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'book_491.npy')])
        elif fold == 2:
            # Train num: 1082 + 41x10 + 72x10 + 491 = 2703
            self.train_npy = 'train_2o5.npy'
            self.train_file = np.load('../user_data/flist/' + self.train_npy)
            for _ in range(10):
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 's1_fake_41.npy')])
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'online_72.npy')])
            self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'book_491.npy')])
        elif fold == 3:
            # Train num: 1198 + 92x10 + 72x10 + 500 = 3338
            self.train_npy = 'train_1o5.npy'
            self.train_file = np.load('../user_data/flist/' + self.train_npy)
            for _ in range(10):
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 's1_fake_92.npy')])
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'online_72.npy')])
            self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'book_weiwei_34.npy')])
        elif fold == 4:
            # Train num: 1082 + 92x10 + 72x10 + 500 = 3222
            self.train_npy = 'train_2o5.npy'
            self.train_file = np.load('../user_data/flist/' + self.train_npy)
            for _ in range(10):
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 's1_fake_92.npy')])
                self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'online_72.npy')])
            self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist/' + 'book_weiwei_34.npy')])

        if fold == 1 or fold == 3:
            self.val_npy = 'val_1o5_256.npy'
        else:
            self.val_npy = 'val_2o5_256.npy'
        self.val_file = np.load('../user_data/flist/' + self.val_npy)

        self.train_num = len(self.train_file)
        self.val_num = len(self.val_file)
        train_dataset = GIID_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = GIID_Dataset(self.val_num, self.val_file, choice='val')

        self.giid_model = GIID_Model().cuda()
        self.n_epochs = 100000000
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=False, num_workers=4)

    def train(self):
        with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
            f.write('\nTrain/Val with ' + self.train_npy + '/' + self.val_npy)
        cnt, gen_losses, f1, iou = 0, [], [], []
        tmp_epoch = 0
        best_score = train_fold_best_val_record[self.fold]
        for epoch in range(1, self.n_epochs):
            for items in self.train_loader:
                cnt += self.batch_size
                self.giid_model.train()
                Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, gen_loss = self.giid_model.process(Ii, Mg)
                self.giid_model.backward(gen_loss)
                gen_losses.append(gen_loss.item())
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                Mo[Mo < 127.5] = 0
                Mo[Mo >= 127.5] = 255
                a, b = metric(Mo / 255, Mg / 255)
                f1.append(a)
                iou.append(b)
                print('Tra (%d/%d): G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                      % (cnt, self.train_num, np.mean(gen_losses), np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou)), end='\r')
                if cnt >= 10000:
                    val_gen_loss, val_f1, val_iou = self.val()
                    print('Val (%d/%d): G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                          % (cnt, self.train_num, val_gen_loss, val_f1, val_iou, val_f1 + val_iou))
                    tmp_epoch = tmp_epoch + 1
                    self.giid_model.save('model_history_fold_%d' % self.fold + '/tmp_epoch_%03d/' % tmp_epoch)
                    if np.mean(val_f1) + np.mean(val_iou) > best_score and tmp_epoch >= 5:
                        best_score = np.mean(val_f1) + np.mean(val_iou)
                        train_fold_best_val_record[self.fold] = best_score
                        np.save('train_fold_best_val_record.npy', train_fold_best_val_record)
                        self.giid_model.save('best_fold_%d/' % self.fold)
                    with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
                        f.write('\n(%d/%d): Tra: G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f Val: G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                                % (cnt, self.train_num, np.mean(gen_losses), np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou), val_gen_loss, val_f1, val_iou, val_f1 + val_iou))
                    cnt, gen_losses, f1, iou = 0, [], [], []
                    if (self.fold == 1 or self.fold == 2) and tmp_epoch >= 18:
                        exit()
                    if tmp_epoch >= 23:
                        exit()

    def val(self):
        self.giid_model.eval()
        f1, iou, gen_losses = [], [], []
        rm_and_make_dir('../user_data/res/val_decompose_256_' + gpu_ids[0] + '/')
        for cnt, items in enumerate(self.val_loader):
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1]
            Mo, gen_loss = self.giid_model.process(Ii, Mg)
            gen_losses.append(gen_loss.item())
            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg), self.convert2(Mo)
            N, H, W, _ = Mg.shape
            Mo[Mo < 127.5] = 0
            Mo[Mo >= 127.5] = 255
            for i in range(len(Mo)):
                cv2.imwrite('../user_data/res/val_decompose_256_' + gpu_ids[0] + '/' + filename[i][:-4] + '.png', Mo[i][..., ::-1])
        val_list = merge(val_npy=self.val_npy)
        path_pre = '../user_data/res/val_merge_256_' + gpu_ids[0] + '/'
        path_gt = '../s2_data/data/train_mask/'
        for file in val_list:
            if '.jpg' in file:
                file = file[:-4] + '.png'
            pre = cv2.imread(path_pre + file)
            gt = cv2.imread(path_gt + file)
            a, b = metric(pre / 255, gt / 255)
            f1.append(a)
            iou.append(b)
        # print('F1:%5.4f, IOU:%5.4f, SUM:%5.4f ' % (np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou)))
        return np.mean(gen_losses), np.mean(f1), np.mean(iou)

    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return img

    def convert2(self, x):
        x = x * 255.
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()


# Predict
def forensics_test(model, s2_path, split_list, size, batch_size, fold, tta_idx, res_path):
    test_path = s2_path + 'test_decompose_%d/' % size
    test_dataset = GIID_Dataset(choice='test', test_path=test_path, tta_idx=tta_idx)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnt = 0
    path_out = res_path + 'fold_%d_tta_%d/test_decompose_%d/' % (fold, tta_idx, size)
    rm_and_make_dir(path_out)
    for items in test_loader:
        cnt += batch_size
        Ii, Mg = (item.cuda() for item in items[:-1])
        filename = items[-1]
        Mo = model(Ii)
        Mo = Mo * 255.
        Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
        for i in range(len(Mo)):
            Mo_tmp = Mo[i][..., ::-1]
            if tta_idx == 5:
                Mo_tmp = cv2.flip(Mo_tmp, 0)
            if tta_idx == 2 or tta_idx == 6:
                Mo_tmp = cv2.rotate(Mo_tmp, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif tta_idx == 3 or tta_idx == 7:
                Mo_tmp = cv2.rotate(Mo_tmp, cv2.cv2.ROTATE_90_CLOCKWISE)
            elif tta_idx == 4 or tta_idx == 8:
                Mo_tmp = cv2.rotate(Mo_tmp, cv2.cv2.ROTATE_180)
            cv2.imwrite(path_out + filename[i][:-4] + '.png', Mo_tmp)
    forensics_test_merge(split_list=split_list, s2_path=s2_path, path_in=path_out,
                         path_out=res_path + 'fold_%d_tta_%d/test_merge_%d/' % (fold, tta_idx, size), size=size)


# Merge the predicted images
def forensics_test_merge(split_list, s2_path, path_in, path_out, size):
    rm_and_make_dir(path_out)
    for file in split_list:
        img = cv2.imread(s2_path + 'test/' + file)
        H, W, _ = img.shape
        X, Y = H // size + 1, W // size + 1
        idx = 0
        rtn = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
                rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] = img_tmp
                idx += 1
            img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
            rtn[x * size: (x + 1) * size, -size:, :] = img_tmp
            idx += 1
        for y in range(Y - 1):
            img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
            rtn[-size:, y * size: (y + 1) * size, :] = img_tmp
            idx += 1
        img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
        rtn[-size:, -size:, :] = img_tmp
        idx += 1
        cv2.imwrite(path_out + file[:-4] + '.png', rtn)


# Decompose the "big" image into several "small" patches
def decompose(s2_path):
    path = s2_path + 'test/'
    flist = sorted(os.listdir(path))
    size_list = [384, 512, 768, 1024]
    for size in size_list:
        path_out = s2_path + 'test_decompose_' + str(size) + '/'
        rm_and_make_dir(path_out)
    rtn_list = [[], [], [], []]
    for file in flist:
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        size_idx = 0
        while size_idx < len(size_list) - 1:
            if H < size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        path_out = s2_path + 'test_decompose_' + str(size) + '/'
        X, Y = H // size + 1, W // size + 1
        idx = 0
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = img[x * size: (x + 1) * size, y * size: (y + 1) * size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size: (x + 1) * size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            img_tmp = img[-size:, y * size: (y + 1) * size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return rtn_list


# Merge the "small" patches back to the "big" image
def merge(val_npy='val_1o5'):
    path = '../s2_data/data/train/'
    path_d = '../user_data/res/val_decompose_256_' + gpu_ids[0] + '/'
    path_r = '../user_data/res/val_merge_256_' + gpu_ids[0] + '/'
    rm_and_make_dir(path_r)
    size = 256

    if val_npy[:7] == 'val_1o5':
        val_list = val_fold_1
    else:
        val_list = val_fold_2

    for file in val_list:
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        X, Y = H // size + 1, W // size + 1
        idx = 0
        rtn = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
                rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] = img_tmp
                idx += 1
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
            rtn[x * size: (x + 1) * size, -size:, :] = img_tmp
            idx += 1
        for y in range(Y - 1):
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
            rtn[-size:, y * size: (y + 1) * size, :] = img_tmp
            idx += 1
        img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
        rtn[-size:, -size:, :] = img_tmp
        idx += 1
        cv2.imwrite(path_r + file[:-4] + '.png', rtn)

    return val_list


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test', 'val'])
    parser.add_argument('--func', type=int, default=0, help='test mode, one of [0, 1, 2] (\'decompose\', \'inference\', \'merge\')')
    parser.add_argument('--size_idx', type=int, default=0, help='index of [384, 512, 768, 1024] (input size)')
    parser.add_argument('--fold', type=int, default=1, help='one of [1, 2, 3, 4] (fold id)')
    parser.add_argument('--tta', type=int, default=1, help='one of [1, ..., 8] (TTA type)')
    args = parser.parse_args()

    if args.type == 'train':
        model = ForgeryForensics(args.fold)
        model.train()
    elif args.type == 'test':
        size_list = [384, 512, 768, 1024]
        batch_list = [16, 12, 6, 6]
        s2_path = '../s2_data/data/'
        res_path = '../user_data/res/'
        if args.func == 0:
            split_list = decompose(s2_path=s2_path)
            np.save(s2_path + 'split_flist.npy', np.array(split_list))
            for fold_i in [1, 2, 3, 4]:
                for tta_i in [1]:
                    rm_and_make_dir(res_path + 'fold_%d_tta_%d/' % (fold_i, tta_i))
            print('Finish decompose test images.')
        elif args.func == 1:
            split_list = np.load(s2_path + 'split_flist.npy', allow_pickle=True)
            if args.fold == 1:
                model_path = 'best_fold_1/'
                # model_path = 'best_fold_1_backup/'
            elif args.fold == 2:
                model_path = 'best_fold_2/'
                # model_path = 'best_fold_2_backup/'
            elif args.fold == 3:
                model_path = 'best_fold_3/'
                # model_path = 'best_fold_3_backup/'
            elif args.fold == 4:
                model_path = 'best_fold_4/'
                # model_path = 'best_fold_4_backup/'
            model = GIID_Model().cuda()
            model.load(model_path)
            model.eval()
            if len(split_list[args.size_idx]):
                forensics_test(model=model, s2_path=s2_path, split_list=split_list[args.size_idx],
                               size=size_list[args.size_idx], batch_size=batch_list[args.size_idx],
                               fold=args.fold, tta_idx=args.tta, res_path=res_path)
            print('Finish inference size %d.' % size_list[args.size_idx])
        else:
            path_out = '../prediction_result/images/'
            rm_and_make_dir(path_out)
            path_list = []
            for fold_i in [1, 2, 3, 4]:
                for tta_i in [1]:
                    path_list.append(res_path + 'fold_%d_tta_%d/' % (fold_i, tta_i))
            for size in size_list:
                if not os.path.exists(path_list[0] + 'test_merge_%d' % size):
                    continue
                file_list = sorted(os.listdir(path_list[0] + 'test_merge_%d' % size))
                for file in file_list:
                    imgs = []
                    for path in path_list:
                        imgs.append(cv2.imread(path + 'test_merge_%d/' % size + file))
                    rtn = imgs[0] / len(imgs)
                    for i in range(1, len(imgs)):
                        rtn += imgs[i] / len(imgs)
                    rtn[rtn < 127.5] = 0
                    rtn[rtn >= 127.5] = 255
                    cv2.imwrite(path_out + file, np.uint8(rtn))
            print('Finish generate final results.')
    elif args.type == 'val':
        model = ForgeryForensics()
        model.val()
