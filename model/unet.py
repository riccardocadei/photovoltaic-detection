import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2*32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*32, out_channels=2*32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=2*32, out_channels=2*64,kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*64, out_channels=2*64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=2*64, out_channels=2*128,kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*128, out_channels=2*128,kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=2*128, out_channels=2*256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*256, out_channels=2*256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=2*256, out_channels=2*512, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*512, out_channels=2*512, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=2*512, out_channels=2*256, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=2*512, out_channels=2*256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*256, out_channels=2*256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=2*256, out_channels=2*128, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=2*256, out_channels=2*128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*128, out_channels=2*128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=2*128, out_channels=2*64, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=2*128, out_channels=2*64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*64, out_channels=2*64,kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=2*64, out_channels=2*32, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=2*64, out_channels=2*32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*32, out_channels=2*32,kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Final output
        self.conv_final = nn.Conv2d(in_channels=2*32, out_channels=1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)

        # Down 2
        x = self.conv2_block(x)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)

        # Down 3
        x = self.conv3_block(x)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)

        # Down 4
        x = self.conv4_block(x)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)

        # Midpoint
        x = self.conv5_block(x)

        # Up 1
        x = self.up_1(x)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        x = self.conv_up_1(x)

        # Up 2
        x = self.up_2(x)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        x = self.conv_up_2(x)

        # Up 3
        x = self.up_3(x)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        x = self.conv_up_3(x)

        # Up 4
        x = self.up_4(x)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        x = self.conv_up_4(x)

        # Final output
        x = self.conv_final(x)

        return x

