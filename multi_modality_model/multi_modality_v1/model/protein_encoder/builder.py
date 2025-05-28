from ....cstp_v3 import modelling

def build_protein_encoder(ckpt):
    esm_embedding = modelling.ProteinSeqEmbeddingExtractor(ckpt)

    return esm_embedding



