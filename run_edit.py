import os

import cv2
import numpy as np
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
import json

from tqdm import tqdm

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    parser.add_argument("--source_experiment",type=str,help="experiment name")
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--feature_injection_threshold',type=int,default=40,help='injection')
    opt = parser.parse_args()
    exp_path_root="./experiment/pnp/"
    with open(exp_path_root+opt.source_experiment+"args.json", "r") as f:
        args = json.load(f)
        seed = args["seed"]
        source_prompt = args["prompt"]
    seed_everything(seed)
    #pnp预测steps
    possible_ddim_steps = args["save_feature_timesteps"]
    assert opt.steps in possible_ddim_steps or opt.steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {opt.steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {opt.steps}"
    ddim_steps = opt.steps if opt.steps is not None else possible_ddim_steps[-1]
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):#从txt中读取prompt
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]
    print(image_paths)

    # prepare models
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))#选取cond——model

    process_cond_module = getattr(api, f'get_cond_{which_cond}')#选取api中对应的get_cond函数

    def load_target_features():
        #选择injection的块索引
        # self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
        self_attn_output_block_indices =[4,5,6,7,8,9,10,11]
        # out_layers_output_block_indices = [4,5,6,7,8,9,10,11]
        out_layers_output_block_indices =[7,8,9,10,11]
        # 设置了固定的self_atten阈限吧
        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [opt.feature_injection_threshold]
        target_features = []

        source_experiment_out_layers_path = os.path.join(exp_path_root, opt.source_experiment, "feature_maps")
        source_experiment_qkv_path = os.path.join(exp_path_root, opt.source_experiment, "feature_maps")
        
        time_range = np.flip(sampler.ddim_timesteps)#反转timesteps作为之后迭代的顺序
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        # 对应地，加载feature_extraction保存的map。t对所有时间步遍历,i则是划分layer
        for i, t in enumerate(iterator):
            current_features = {}
            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                if i <= int(output_block_self_attn_map_injection_threshold):
                    output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                    output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                if i <= int(feature_injection_threshold):
                    output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                    current_features[f'output_block_{output_block_idx}_out_layers'] = output

            target_features.append(current_features)

        return target_features

    injected_features = load_target_features()
    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)#即用对应的get_cond处理image
                # 保存生成的condition
                base_count = len(os.listdir(opt.outdir)) // 2
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))
                #
                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context,injected_features=injected_features,ddim_steps=ddim_steps)#推理
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


if __name__ == '__main__':
    main()
