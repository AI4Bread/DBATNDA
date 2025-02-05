import torch
import numpy as np
import pandas as pd


#dataset1
def read_file1():
    id_lncdi = np.loadtxt(".../dataset1/id_lncdi.txt")
    id_milnc = np.loadtxt(".../dataset1/id_milnc.txt")
    id_midi = np.loadtxt(".../dataset1/id_midi.txt")
    mm_sim_dict_lnc = np.load(
        '.../processed_data/dataset1/lnc_fun.npy')
    dd_sim_dict = np.load(
        '.../processed_data/dataset1/dis_fun.npy')
    mm_sim_dict_mi = np.load(
        '.../processed_data/dataset1/mi_fun.npy')

    lncrna_data_dict_LDA = np.load(
        ".../processed_data/dataset1/lnc_dis.npy")
    disease_data_dict_LDA = np.load(
        ".../processed_data/dataset1/dis_lnc.npy")

    mirna_data_dict_MDA = np.load(
        ".../processed_data/dataset1/mi_dis.npy")
    disease_data_dict_MDA = np.load(
        ".../processed_data/dataset1/dis_mi.npy")

    mirna_data_dict_LMI = np.load(
        ".../processed_data/dataset1/mi_lnc.npy")
    lncrna_data_dict_LMI = np.load(
        ".../processed_data/dataset1/lnc_mi.npy")

    lnc_mi = np.loadtxt(".../dataset1/yuguoxian_lnc_mi.txt")
    mi_lnc = lnc_mi.T
    lnc_dis = np.loadtxt(".../dataset1/lnc_dis_association.txt")
    mi_dis = np.loadtxt(".../dataset1/mi_dis.txt")


    return (id_lncdi, id_milnc, id_midi, mm_sim_dict_lnc, dd_sim_dict, mm_sim_dict_mi, lncrna_data_dict_LDA,
                disease_data_dict_LDA, mirna_data_dict_MDA, disease_data_dict_MDA, mirna_data_dict_LMI, lncrna_data_dict_LMI,
                lnc_dis, mi_dis, mi_lnc)

#dataset2
def read_file2():
    id_lncdi = np.loadtxt(".../dataset2/id_lncdi.txt")
    id_milnc = np.loadtxt(".../dataset2/id_milnc.txt")
    id_midi = np.loadtxt(".../dataset2/id_midi.txt")
    di_lnc = pd.read_csv('.../dataset2/di_lnc_intersection.csv',
                         index_col='Unnamed: 0')
    di_mi = pd.read_csv('.../dataset2/di_mi_intersection.csv',
                        index_col='Unnamed: 0')
    mi_lnc = pd.read_csv('.../dataset2/mi_lnc_intersection.csv',
                         index_col='Unnamed: 0')

    lnc_dis = di_lnc.values.T
    mi_dis = di_mi.values.T
    mi_lnc = mi_lnc.values

    mm_sim_dict_lnc = np.load(
        '.../processed_data/dataset2/lnc_fun.npy')
    dd_sim_dict = np.load(
        '.../processed_data/dataset2/dis_fun.npy')
    mm_sim_dict_mi = np.load(
        '.../processed_data/dataset2/mi_fun.npy')

    lncrna_data_dict_LDA = np.load(
        ".../processed_data/dataset2/lnc_dis.npy")
    disease_data_dict_LDA = np.load(
        ".../processed_data/dataset2/dis_lnc.npy")

    mirna_data_dict_MDA = np.load(
        ".../processed_data/dataset2/mi_dis.npy")
    disease_data_dict_MDA = np.load(
        ".../processed_data/dataset2/dis_mi.npy")

    mirna_data_dict_LMI = np.load(
        ".../processed_data/dataset2/mi_lnc.npy")
    lncrna_data_dict_LMI = np.load(
        ".../processed_data/dataset2/lnc_mi.npy")
    return (id_lncdi, id_milnc, id_midi, mm_sim_dict_lnc, dd_sim_dict, mm_sim_dict_mi, lncrna_data_dict_LDA,
                disease_data_dict_LDA, mirna_data_dict_MDA, disease_data_dict_MDA, mirna_data_dict_LMI, lncrna_data_dict_LMI,
                lnc_dis, mi_dis, mi_lnc)

def get_feature(sim_dict1, sim_dict2, data_dict1, data_dict2, index, adi_matrix, category):
    CATEGORY_SPACE = ["lncdi", "midi", "milnc"]
    assert category in CATEGORY_SPACE
    if category == 'lncdi':
        label_c = 0
    elif category == 'midi':
        label_c = 1
    else:
        label_c = 2
    input = []
    output = []
    output_2 = []
    for i in range(index.shape[0]):
        cate1 = int(index[i][0])
        cate2 = int(index[i][1])
        ver1 = sim_dict1[cate1].tolist() + data_dict1[cate1].tolist()
        ver2 = sim_dict2[cate2].tolist() + data_dict2[cate2].tolist()
        ver = ver1 + ver2
        input.append(ver)
        label = (3 * adi_matrix[[cate1], [cate2]] + label_c).tolist() #multi-class
        label_2 = (adi_matrix[[cate1], [cate2]]).tolist() #binary-class
        output.append(label)
        output_2.append(label_2)
    output = np.array(output)
    output = output.ravel()
    output_2 = np.array(output_2)
    output_2 = output_2.ravel()
    return np.array(input), output, output_2

# train-test split for different negative sample ratio
def create_train_test_sets_cv(labels, negative_ratio):
    positive_indices = np.argwhere(labels > 2)
    np.random.seed(0)
    np.random.shuffle(positive_indices)

    negative_indices = np.argwhere(labels <= 2)

    positive_subsets = np.array_split(positive_indices, 5)

    test_positive_indices = positive_subsets[0]
    train_positive_indices = np.concatenate(positive_subsets[1:])
    if isinstance(negative_ratio, (int, float)):
        num_test_negatives = int(negative_ratio * len(test_positive_indices))
    else:
        num_test_negatives = int(len(negative_indices))
    train_negative_indices = negative_indices[
        np.random.choice(len(negative_indices), len(train_positive_indices), replace=False)]
    test_negative_indices = negative_indices[
        np.random.choice(len(negative_indices), num_test_negatives, replace=False)]

    train_mask = np.vstack((train_positive_indices, train_negative_indices))
    test_mask = np.vstack((test_positive_indices, test_negative_indices))

    return train_mask, test_mask

# edge_index
def create_eindex(labels, index, num_c):
    source_node = []
    target_node = []
    num_index = []
    for i in range(labels.shape[0]):
        if labels[i] > 2:
            source_node.append(int(index[i][0]))
            target_node.append(int(index[i][1]))
            num_index.append(i + num_c)
    edge_index = torch.tensor([source_node, target_node], dtype=torch.long)
    return edge_index, num_index     # original edge index

def create_newindex(ei1, nindex1, x, ei2, nindex2, y, edge_index):
    for i in range(ei1.shape[1]):
        for j in range(ei2.shape[1]):
            if ei1[x][i] == ei2[y][j] :
                edge_index.append([nindex1[i], nindex2[j]])
    return edge_index


if __name__ == '__main__':
    (id_lncdi, id_milnc, id_midi, mm_sim_dict_lnc, dd_sim_dict, mm_sim_dict_mi, lncrna_data_dict_LDA,
             disease_data_dict_LDA, mirna_data_dict_MDA, disease_data_dict_MDA, mirna_data_dict_LMI, lncrna_data_dict_LMI,
             lnc_dis, mi_dis, mi_lnc)=read_file1()
    '''(id_lncdi, id_milnc, id_midi, mm_sim_dict_lnc, dd_sim_dict, mm_sim_dict_mi, lncrna_data_dict_LDA,
     disease_data_dict_LDA, mirna_data_dict_MDA, disease_data_dict_MDA, mirna_data_dict_LMI, lncrna_data_dict_LMI,
     lnc_dis, mi_dis, mi_lnc) = read_file2()'''

    features_LDA, labels_lncdi, labels_lncdi2 = get_feature(mm_sim_dict_lnc, dd_sim_dict, lncrna_data_dict_LDA, disease_data_dict_LDA,
                                         id_lncdi, lnc_dis, category='lncdi')
    features_MDA, labels_midi, labels_midi2 = get_feature(mm_sim_dict_mi, dd_sim_dict, mirna_data_dict_MDA, disease_data_dict_MDA,
                                         id_midi, mi_dis, category='midi')
    features_LMI, labels_milnc, labels_milnc2 = get_feature(mm_sim_dict_mi, mm_sim_dict_lnc, mirna_data_dict_LMI, lncrna_data_dict_LMI,
                                         id_milnc, mi_lnc, category='milnc')

    #features
    node_features = np.vstack((features_LDA, features_MDA, features_LMI))
    node_features = torch.from_numpy(node_features)
    print('features done')

    #labels
    labels = np.hstack((labels_lncdi, labels_midi, labels_milnc))
    print('labels_done')

    #labels2
    labels2 = np.hstack((labels_lncdi2, labels_midi2, labels_milnc2))
    print('labels2_done')

    #edge index
    edge_index = []
    ei_lindi, nindex_lindi = create_eindex(labels_lncdi, id_lncdi, 0)
    ei_midi, nindex_midi = create_eindex(labels_midi, id_midi, id_lncdi.shape[1])
    ei_milnc, nindex_milnc = create_eindex(labels_milnc, id_milnc, id_lncdi.shape[1] + id_midi.shape[1])

    edge_index = create_newindex(ei_lindi, nindex_lindi, 1, ei_midi, nindex_midi, 1, edge_index)
    edge_index = create_newindex(ei_lindi, nindex_lindi, 0, ei_milnc, nindex_milnc, 1, edge_index)
    edge_index = create_newindex(ei_midi, nindex_midi, 0, ei_milnc, nindex_milnc, 0, edge_index)

    print('edge_index_done')

    train_mask, test_mask = create_train_test_sets_cv(labels, negative_ratio=10)
    train_mask = np.squeeze(train_mask)
    test_mask = np.squeeze(test_mask)

    labels = torch.from_numpy(labels)
    labels2 = torch.from_numpy(labels2)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    train_mask = torch.from_numpy(train_mask)
    test_mask = torch.from_numpy(test_mask)

    print(train_mask.shape)
    print(test_mask.shape)


    torch.save(node_features,
               ".../node_features.pt")
    torch.save(edge_index,
               '.../edge_index.pt')
    torch.save(labels,
               ".../labels.pt")
    torch.save(labels2,
               ".../labels_2.pt")
    torch.save(train_mask, ".../10/train_mask.pt")
    torch.save(test_mask, ".../10/test_mask.pt")