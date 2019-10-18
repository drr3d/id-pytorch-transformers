import torch
import os

def restoreModel(model, resume_iters, model_name, model_save_dir, is_finetune=False, base_model_prefix='xlnet', from_pretrained=False):
    """Restore the trained generator and discriminator."""

    model_path = os.path.join(model_save_dir, '{}.ckpt'.format(model_name))
    print("restoreModel model_path: {}".format(model_path))
    if resume_iters is not None:
        print('Loading the trained models from step {} - of file: {}'.format(resume_iters, model_path))

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    if from_pretrained:
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, base_model_prefix) and any(s.startswith(base_model_prefix) for s in state_dict.keys()):
            start_prefix = base_model_prefix + '.'
        if hasattr(model, base_model_prefix) and not any(s.startswith(base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
        print("loading_info: {}".format(loading_info))

    else:
        # mainly for resume training
        model.load_state_dict(state_dict)
    print("Loading pytorch previous model done...")
    if is_finetune:
        model.zero_grad()
    return model