# Embedding without forward pass
        # Disabled : computational graph requires forward pass     
        # for t in range(0, num_triplets, bsize):
        #     anchor = Variable(torch.FloatTensor(np.take(embedding, triplets[t:t+bsize, 0].tolist(), axis=0).astype(np.float32)).cuda(),
        #                       requires_grad=True)
        #     positive = Variable(torch.FloatTensor(np.take(embedding, triplets[t:t+bsize, 1].tolist(), axis=0).astype(np.float32)).cuda(),
        #                         requires_grad=True)
        #     negative = Variable(torch.FloatTensor(np.take(embedding, triplets[t:t+bsize, 2].tolist(), axis=0).astype(np.float32)).cuda(),
        #                         requires_grad=True)
