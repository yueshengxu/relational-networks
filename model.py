import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#four convolutional layers with 32, 64, 128 and 256 kernels, 
#ReLU non-linearities, and batch normalization using parameters from original paper. 
#
#However, we found reducing the number of nodes would not cause a big drop in accuracy
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)
        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

#Last fully connected layer with a softmax output
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()
        # self.fc2 = nn.Linear(256, 256) #        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)  #        self.fc3 = nn.Linear(256, 10)
        

        
    def forward(self, x):
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_state, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_state, input_qst)
        #loss = F.nll_loss(output, label)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_state, input_qst, label):
        output = self(input_img, input_state, input_qst)
        loss = F.cross_entropy(output, label)
        #loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))
        
class RN2(BasicModel):
    def __init__(self, args):
        print("RN2")
        super(RN2, self).__init__(args, 'RN2')
        self.conv = ConvInputModel()
        self.relation_type = args.relation_type
        

        # state description, itself is a representation of image
        # we match output shape 7*2+11: 
        # 7 = number of feature of one object
        # 11= length of question embeddings
        # 7*2 because we pair each two objects together
        if args.state_desc == 'state-desc':
            print("state-desc")
            self.g_fc1 = nn.Linear(7*2+11, 1000)
        #if we use pixel version, idea is similar expect we change different shape
        else:
            print("pixel")
            self.g_fc1 = nn.Linear((256+2)*2+11, 1000)


        
        # 4 layers of MLP for g 
        # much with less nodes than paper proposed
        # But we are still able to reach accuracy above 90   
        self.g_fc2 = nn.Linear(1000, 1000)
        self.g_fc3 = nn.Linear(1000, 1000)
        self.g_fc4 = nn.Linear(1000, 1000)
        # 3 layers of MLP for f
        self.f_fc1 = nn.Linear(1000, 500)
        self.f_fc2 = nn.Linear(500, 256)
        self.f_fc3 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        
        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, state, qst):
        if state == None:
            x = self.conv(img) ## x = (64 x 24 x 5 x 5)
            """g"""
            mb = x.size()[0]
            n_channels = x.size()[1]
            d = x.size()[2]
            # x_flat = (64 x 25 x 24)
            x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
            
            # add coordinates
            x_flat = torch.cat([x_flat, self.coord_tensor],2)
            
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)  #(64x1x11)
            qst = qst.repeat(1, 25, 1)     #(64x25x11)
            qst = torch.unsqueeze(qst, 2)  #(64x25x1x11)
            
            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)   # (64x1x25x26+18)
            x_i = x_i.repeat(1, 25, 1, 1)      # (64x25x25x26+18)
            x_j = torch.unsqueeze(x_flat, 2)   # (64x25x1x26+18)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
            
            x_ = x_full.view(mb * (d * d) * (d * d), 527)  # (64*25*25x2*26*18) = (40.000, 70)
        else:
            x_flat = state
            # x_flat = state.reshape((64,7,6))
            
            mb = x_flat.size()[0]
            n_channels = x_flat.size()[1]
            d = x_flat.size()[2]
            
            


            # add question everywhere
            qst = torch.unsqueeze(qst, 1) # (64x1x11)
            qst = qst.repeat(1, 6, 1)     # (64x6x11)
            qst = torch.unsqueeze(qst, 2) # (64x6x1x11)


            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x6x7)
            x_i = x_i.repeat(1, 6, 1, 1)      # (64x6x6x7)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x6x1x7)

            x_j = torch.cat([x_j, qst], 3)    # (64x6x1x18)
            x_j = x_j.repeat(1, 1, 6, 1)      # (64x6x6x18)

            # concatenate all together
            x_full = torch.cat([x_i,x_j],3)   # (64x6x6x25)
    
            # reshape for passing through network



            x_ = x_full.view(64*6*6, 7*2+11)  
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        

        if state == None:
            x_g = x_.view(mb, (d * d) * (d * d), 1000)
        else: 
            x_g = x_.view(mb, 6*6, 1000)
        
        #print(x_g.shape)
        x_g = x_g.sum(1).squeeze()
        #print(x_g.shape)
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)
        x_f = F.relu(x_f)
        
        
        
        return self.fcout(x_f)



############################################################################################
#####                        Model RN and CNN_MLP are not used
class RN(BasicModel):
    def __init__(self, args):
        print("RN")
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type

        #(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)
        #self.g_fc1 = nn.Linear((256+2)*2+18, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)



        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        # print(x.shape)
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, 25, 1)
        qst = torch.unsqueeze(qst, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
        x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
    
        # reshape for passing through network
        #print("Xfull",x_full.shape)
        x_ = x_full.view(mb * (d * d) * (d * d), 70)  # (64*25*25x2*26*18) = (40.000, 70)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        


        x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)
class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)

