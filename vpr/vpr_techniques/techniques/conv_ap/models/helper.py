import numpy as np
from vpr.vpr_techniques.techniques.conv_ap.models import aggregators, backbones



def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[], ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where we sometimes need to crop the last residual block (ex. [4]). Defaults to [].

    Returns:
        model: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze + 2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained,
                                          layers_to_freeze=layers_to_freeze)

    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k',
                              pretrained=pretrained,
                              layers_to_freeze=layers_to_freeze)


def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """

    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)

    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)


# -------------------------------------
def print_nb_params(m):
    """Prints the numbe of trainable parameters in the model

    Args:
        m (nn.Module): PyTorch model
    """
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params / 1e6:.3}M')


def main():
    import torch

    x = torch.randn(1, 3, 224, 224)  # random image
    # backbone = get_backbone(backbone_arch='resnet50')
    backbone = get_backbone(backbone_arch='resnet50')
    agg = get_aggregator('cosplace', {'in_dim': backbone.out_channels, 'out_dim': 512})
    # agg = get_aggregator('GeM')
    print_nb_params(backbone)
    print_nb_params(agg)

    backbone_output = backbone(x)
    agg_output = agg(backbone_output)
    print(f'output shape: {agg_output.shape}')


if __name__ == '__main__':
    main()