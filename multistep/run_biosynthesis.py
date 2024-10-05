import os
import sys
import time

import pynvml
from rdkit import Chem

from retro_star.api import RSPlanner


def get_avai_gpu():
    pynvml.nvmlInit()
    for gpu_id in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / meminfo.total > 0.2:
            return gpu_id
    return -1


def save_txt(data, path):
    with open(path, 'w') as f:
        for each in data:
            f.write(each + '\n')


def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data


def run(input_dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    # canonicalization
    mol = Chem.MolToSmiles(Chem.MolFromSmarts(input_dict['target_mol']))
    one_step_model_path = [
        '../singlestep/checkpoints/np-like/model_step_30000.pt',
        '../singlestep/checkpoints/np-like/model_step_50000.pt',
        '../singlestep/checkpoints/np-like/model_step_80000.pt',
        '../singlestep/checkpoints/np-like/model_step_100000.pt'
    ]
    value_fn_model_path = './retro_star/saved_models/best_epoch_final_4.pt'
    viz_dir = os.path.join('viz' + str(int(time.time())) + '_' + input_dict['target_mol_name'])
    ret_file_path = os.path.join('./viz/tmp/', viz_dir + '.zip')
    planner = RSPlanner(
        gpu=get_avai_gpu(),
        use_value_fn=True,
        value_fn_model_path=value_fn_model_path,
        fp_dim=2048,
        iterations=input_dict['expansion_iters'],
        expansion_topk=input_dict['expansion_topk'],
        route_topk=input_dict['route_topk'],
        one_step_model_type='onmt',
        buliding_block_path=input_dict['building_blocks'],
        mlp_templates_path=None,
        one_step_model_path=one_step_model_path,
        beam_size=20,
        viz=True,
        viz_dir=viz_dir
    )

    result = planner.plan(mol)

    return result


def get_input():
    input_data = {
        'target_mol': 'N[C@@H](CNC(=O)C(=O)O)C(=O)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
    }
    print('Input: ')
    target_mol = sys.stdin.readline().strip('\n')
    input_data['target_mol'] = target_mol
    return input_data


def main_biosynthesis():
        
    input_dicts = [
        {'target_mol_name': 'Lycosantalonol',
        'target_mol': 'CC(=CCCC(C)(C(=O)CCC1(C2CC3C1(C3C2)C)C)O)C',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Avenacin_A1',
        'target_mol': 'C[C@]12CC[C@@H]([C@@]([C@@H]1CC[C@@]3([C@@H]2C[C@@H]4[C@]5([C@]3(C[C@@H]([C@@]6([C@H]5C[C@@]([C@H](C6)OC(=O)C7=CC=CC=C7NC)(C)C=O)C)O)C)O4)C)(C)CO)O[C@H]8[C@@H]([C@H]([C@H](CO8)O[C@H]9[C@@H]([C@H]([C@@H]([C@H](O9)CO)O)O)O)O)O[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Momilactone_B',
        'target_mol': 'C[C@]1(CC[C@@H]2C(=CC3[C@@H]4[C@]25CCC([C@@]4(C(=O)O3)C)(OC5)O)C1)C=C',
        'expansion_topk': 80,
        'max_depth': 20,
        'expansion_iters': 50,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Afrormosin',
        'target_mol': 'COC1=CC=C(C=C1)C2=COC3=CC(=C(C=C3C2=O)OC)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Vincamine',
        'target_mol': 'CC[C@@]12CCCN3[C@@H]1C4=C(CC3)C5=CC=CC=C5N4[C@](C2)(C(=O)OC)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Brucine',
        'target_mol': 'COC1=C(C=C2C(=C1)[C@]34CCN5[C@H]3C[C@@H]6[C@@H]7[C@@H]4N2C(=O)C[C@@H]7OCC=C6C5)OC',
        'expansion_topk': 100,
        'max_depth': 100,
        'expansion_iters': 200,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Diaboline',
        'target_mol': 'CC(=O)N1[C@H]2[C@H]3[C@H]4C[C@H]5[C@@]2(CCN5CC4=CCO[C@H]3O)C6=CC=CC=C61',
        'expansion_topk': 80,
        'max_depth': 20,
        'expansion_iters': 50,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Falcarindiol',
        'target_mol':'CCCCCCC/C=C\\[C@@H](C#CC#C[C@@H](C=C)O)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': 'retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv'
        },

        {'target_mol_name': 'Beta_Amyrin',
        'target_mol':'CC1(C)CC[C@]2(C)CC[C@]3(C)C(=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@]43C)[C@@H]2C1',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        'building_blocks': '/home/lmartins/BioNaviNP_LuciEdition/multistep/retro_star/dataset/bio_data/bio_building_blocks_all/building_block_beta_amyrin.csv'
        }
    ]
    
    target_name = "Beta_Amyrin"
    target_input = next(input_dict for input_dict in input_dicts if input_dict['target_mol_name'] == target_name)
    
    res = run(target_input)
    print(res)

    # for input_dict in input_dicts:
    #     res = run(input_dict)
    #     print(res)


if __name__ == '__main__':
    main_biosynthesis() 
