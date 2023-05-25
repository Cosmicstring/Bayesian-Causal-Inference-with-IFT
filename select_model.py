from indipendent_model import Indipendent_model
from confounder_model import Confounder_model_v1, Confounder_model_v2, \
    Confounder_model_v3, Confounder_model_v4, Confounder_model_v5
from bipartite_model import Bipartite_model, Bipartite_model_v2, Bipartite_model_v3, Bipartite_model_v4

def select_model(cm):

    # NOTE: For the inference directions "X->Y" and "Y->X" one has
    # the inference for lognormal field and the nonlinear response field
    # completely decoupled

    # Assumptions for the domains:
    #
    # X : RGSpace
    # Y : Unstructured Domain
    # F : (nonlinear mapping from X->Y space)
    #       domain == RGSpace, target_domain == UnstructuredDomain
    # Beta : (lognormal field from counts_X -> X)
    #       domain == UnstructuredDomain, target_domain == RGSpace

    if cm.direction == "X->Y" or cm.direction=="Y->X":

        if cm.version == 'v1':
            return Bipartite_model(cm)
        elif cm.version == 'v2':
            return Bipartite_model_v2(cm)
        elif cm.version == 'v3':
            return Bipartite_model_v3(cm)
        elif cm.version == 'v4':
            return Bipartite_model_v4(cm)
        else:
            raise NotImplementedError        

    elif cm.direction == "X || Y":

        return Indipendent_model(cm)

    elif cm.direction == "X<-Z->Y":

        if cm.version == 'v1':
            return Confounder_model_v1(cm)
        elif cm.version == 'v2':
            return Confounder_model_v2(cm)
        elif cm.version == 'v3':
            return Confounder_model_v3(cm)
        elif cm.version == 'v4':
            return Confounder_model_v4(cm)
        elif cm.version == 'v5':
            return Confounder_model_v5(cm)
        else:
            raise NotImplementedError
    else:
        raise ValueError("Not implemented")
