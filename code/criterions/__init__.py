from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .dual_space_kd_new import DualSpaceKDWithCMA_OT
from .ULD_1 import UniversalLogitDistillation_1
from .MultiLevelOT import MultiLevelOT
from .MultiLevelOT_1 import MultiLevelOT_1

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "dual_space_kd_with_cma_ot": DualSpaceKDWithCMA_OT,
    "uld_1": UniversalLogitDistillation_1,
    "MultiLevelOT": MultiLevelOT,
    "MultiLevelOT_1": MultiLevelOT_1
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")