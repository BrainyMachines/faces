## STEP 2 : COMPUTE (APPROXIMATE) NEAREST NEIGHBOURS 

        ## Hard triplet mining with FLANN
        ## Disabled as FAISS is used instead
        # hard = []    
        # ann = [flann.FLANN() for i in range(args.num_identities)]
        # ann_neg = [flann.FLANN() for i in range(args.num_identities)]
        # for k in range(args.num_identities):
        #     args.ann_file = os.path.join(args.ckpt_dir, 'ann_{:s}_{:s}_{:04d}.npz'.format(args.dset_name, args.arch, k))
        #     ann_params = ann[k].build_index(embedding_id[k], algorithm='autotuned', target_precision=0.9, log_level='error')
        #     ann_index, ann_dist = ann[k].nn_index(embedding_id[k], embedding_id[k].shape[0], checks=ann_params['checks'])
        #     ann_neg_params = ann_neg[k].build_index(embedding_neg_id[k], algorithm='autotuned', target_precision=0.9, log_level='error')
        #     ann_neg_index, ann_neg_dist = ann_neg[k].nn_index(embedding_id[k], args.num_neighbors, checks=ann_neg_params['checks'])
        #     for a_ in range(ann_index.shape[0]):
        #         for p_ctr in range(args.num_neighbors):
        #             p_ = int(ann_index.shape[1]) - 1  - p_ctr
        #             a = index_id[k][a_]
        #             for n_ in range(args.num_neighbors):
        #                 p = index_id[k][ann_index[a_, p_]]
        #                 n = index_neg_id[k][ann_neg_index[a_, n_]]
        #                 if ann_dist[a_, p_] - ann_neg_dist[a_, n_] + args.margin >= 0:  # hard example: violates margin 
        #                     hard.append((a, p, n))
        #     print('#Tuples: ', len(hard))
        #     print(hard)
        #     joblib.dump({'ann_index': ann_index, 'ann_dist': ann_dist, 'ann_neg_index': ann_neg_index, 'ann_neg_dist': ann_neg_dist}, args.ann_file)
        #     ann_params = None
        #     ann_index = None
        #     ann_dist = None
        # ann = None

