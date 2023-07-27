from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import config
from vpr.vpr_techniques.techniques.selectCNN.models import resnet9regressionModule
from vpr.vpr_techniques.techniques.selectCNN.selectCNN_config import weights_path, techniques
from vpr.data.utils import VprDataset
import numpy as np
import faiss
from numpy.linalg import norm
from scipy.special import softmax
import torch
from vpr.vpr_techniques.utils import save_descriptors
from torchvision import transforms


NAME = "selectCNN_recallat10"


#selection_model = resnet9regressionModule.load_from_checkpoint(checkpoint_path=weights_path)
#selection_model.eval()

resnet_transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),  # must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def logits2selections(logits: torch.Tensor) -> list:
    selection_probs = logits
    selections = torch.argmax(selection_probs, dim=1).numpy()
    return list(selections.flatten())


def technique_selections(Q: list) -> list:
    ds = VprDataset(Q, transform=resnet_transforms_test)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    all_selections = []
    for batch in dl:
        logits = selection_model(batch.to(config.device)).detach().cpu()
        selections = logits2selections(logits)
        all_selections += selections
    return all_selections


def oracle_selections(Q: list) -> list:
    import pandas as pd
    df1 = pd.read_csv(
        '/home/oliver/Documents/github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/Nordlands_passes_recall@10_test.csv')
    df2 = pd.read_csv(
        '/home/oliver/Documents/github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/SFU_recall@10_test.csv')
    df3 = pd.read_csv(
        '/home/oliver/Documents/github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/StLucia_recall@10_test.csv')
    df4 = pd.read_csv(
        '/home/oliver/Documents/github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/GardensPointWalking_recall@10_test.csv')

    df = pd.concat((df1, df2, df3, df4))
    df_queries = list(df["query_images"].to_numpy().flatten())
    selections = []
    for q in Q:
        idx = df_queries.index(q)
        selections.append(np.argmax(df.iloc[idx].to_numpy()[2:].flatten()))
    return selections



def compute_query_desc(Q: list, dataset_name: str=None, disable_pbar: bool=False) -> tuple[list, list]:
    # Computing VPR technqiue selections
    #selections = technique_selections(Q)
    selections = oracle_selections(Q)

    # Computing the descriptors with VPR method selections
    select_masks = [np.where(np.array(selections)==i) for i in range(len(techniques))]
    select_queries = [np.array(Q)[mask] for mask in select_masks]
    # Compute select descriptors with their corresponding technique
    all_desc = []
    for i, queries in enumerate(select_queries):
        if len(queries) >= 1:
            all_desc.append(techniques[i].compute_query_desc(queries))
        else:
            all_desc.append(np.array([]))

    # compute a list of descriptors that correspond to the queries
    q_desc = []
    counters = np.zeros(len(techniques), dtype=int)
    for i, sel in enumerate(selections):
        q_desc.append(all_desc[sel][counters[sel]])
        counters[sel] += 1

    # saving and returning both query vpr technique selections and descriptors
    all_desc = (q_desc, selections)

    # save the descriptors if a technique name is provided
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='query')

    return all_desc


def compute_map_features(M: list, dataset_name: str=None, disable_pbar: bool=False) -> list:
    # compute all the descriptions with the requied techniques
    all_desc = [technique.compute_map_features(M, disable_pbar=disable_pbar) for technique in techniques]
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='map')
    return all_desc


def matching_method(q_desc: tuple[list, list], m_desc: list[np.ndarray]) -> np.ndarray:
    desc, select_idx = q_desc
    # mask out the descriptors for each technique
    masks = [np.where(np.array(select_idx) == i) for i in range(len(m_desc))]
    # collect the descriptors computed with a particular technique
    select_descriptors = [np.array(desc, dtype=object)[mask] for mask in masks]
    select_descriptors = [np.vstack(d) if len(d) > 0 else np.array([]) for d in select_descriptors]
    # compute the cosine similarity between the technqiue and map descriptors
    seperated_S = [cosine_similarity(d, m_desc[i]) if len(d) != 0 else np.zeros(m_desc[0].shape[0]) for i, d in enumerate(select_descriptors)]
    # put the similarity bectors back into a similarity matrix
    S = np.zeros((len(select_idx), m_desc[0].shape[0]))
    for i, mask in enumerate(masks):
        S[mask] = seperated_S[i]
    # Transpose the similarity matrix to get reference by query orientation
    S = S.transpose()
    # Normalize the similarity matrix
    #S = S/np.linalg.norm(S, axis=0, keepdims=True)
    S = (S - np.mean(S, axis=0, keepdims=True)) / np.std(S, axis=0, keepdims=True)
    #S = softmax(S*0.5, axis=0)
    #S = (S - np.amin(S)) / (np.amax(S) - np.amin(S))
    return S


@torch.no_grad()
class PlaceRecognition:
    def __init__(self, m_desc: list[np.ndarray]):
        self.m_desc = m_desc
        self.indexes = [faiss.IndexFlatIP(m.shape[1]) for m in m_desc]
        for i, desc in enumerate(m_desc):
            faiss.normalize_L2(desc)
            self.indexes[i].add(desc)

    def perform_vpr(self, q_path: list[str]) -> tuple[np.ndarray, np.ndarray]:
        q_desc, selections = compute_query_desc(q_path, disable_pbar=True)
        Is = []
        all_scores = []
        for i, desc in enumerate(q_desc):
            d = desc[None, :].astype(np.float32)
            faiss.normalize_L2(d)
            D, I = self.indexes[selections[i]].search(d, 1)
            temp_mdesc = self.m_desc[selections[i]].squeeze() if self.m_desc[selections[i]].squeeze().ndim == 2 else \
                self.m_desc[selections[i]][0]
            scores = cosine_similarity(q_desc[i][None, :], temp_mdesc).diagonal()
            Is += list(I.flatten())
            all_scores.append(scores)

        return np.array(Is), np.array(all_scores).flatten()

