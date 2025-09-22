import yaml
import os

class InlineListDumper(yaml.SafeDumper):
    '''Determines the formatting of the yaml file'''
    def increase_indent(self, flow=False, indentless=False):
        '''Enforces indentation'''
        return super(InlineListDumper, self).increase_indent(flow, False)

    def represent_list(self, data):
        '''Decides if list are written inline or as a block'''
        if isinstance(data, list) and (all(isinstance(i, int) for i in data) or all(isinstance(i, float) for i in data)):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

def create_config(new_points, n_dim, output_file='output.yaml', prior_type="fixed"):
    '''Function to generate config for the RUN3ModelGen for free ranges or fixed points'''
    base_order = ["M_1", "M_2", "tanb", "mu", "M_3", "AT", "Ab", "Atau",
                  "mA", "mqL3",  "mtR", "mbR", "meL", "mtauL", "meR", "mtauR", 
                  "mqL1", "muR", "mdR"]
    order = {i: base_order[:i] for i in range(1, len(base_order) + 1)}
    selected_params = order.get(n_dim, [])

    # Ranges for free parameters
    free_flat_ranges = {
        "M_1": [-2000, 2000], 
        "M_2": [-2000, 2000],
        "tanb": [1, 60],
        "mu": [-2000, 2000],
        "M_3": [1000, 5000],
        "AT": [-8000, 8000],
        "Ab": [-2000, 2000],
        "Atau": [-2000, 2000],
        "mA": [0, 5000],
        "mqL3": [2000, 5000],
        "mtR": [2000, 5000],
        "mbR": [2000, 5000],
        "meL": [0, 10000],
        "meR": [0, 10000],
        "mtauL": [0, 10000],
        "mtauR": [0, 10000],
        "mqL1": [0, 10000],
        "muR": [0, 10000],
        "mdR": [0, 10000]
    }

    # Ranges for not free parameters
    default_flat_ranges = {
        "M_1": [2000, 2000],
        "M_2": [2000, 2000],
        "mu": [2000, 2000],
        "tanb": [60, 60],
        "M_3": [4000, 4000],  
        "AT": [4000, 4000],
        "Ab": [2000, 2000],
        "Atau": [2000, 2000],
        "mA": [2000, 2000],
        "mqL3": [4000, 4000],
        "mtR": [4000, 4000],
        "mbR": [4000, 4000],
        "meL": [2000, 2000],
        "meR": [2000, 2000],
        "mtauL": [2000, 2000],
        "mtauR": [2000, 2000],
        "mqL1": [4000, 4000],
        "muR": [4000, 4000],
        "mdR": [4000, 4000]
    }

    if prior_type == "flat":

        parameters = {}
        for param in base_order:
            if param in selected_params:
                parameters[param] = free_flat_ranges.get(param, [0, 1])
            else:
                parameters[param] = default_flat_ranges.get(param, [0, 0])

        new_length = new_points

    # For fixed points
    elif prior_type == "fixed":
        parameters = {
            "M_1": [2000],
            "M_2": [2000],
            "tanb": [60],
            "mu": [2000],
            "M_3": [4000],
            "AT": [4000],
            "Ab": [2000],
            "Atau": [2000],
            "mA": [2000],
            "mqL3": [4000],
            "mtR": [4000],
            "mbR": [4000],
            "meL": [2000],
            "mtauL": [2000],
            "meR": [2000],
            "mtauR": [2000],
            "mqL1": [4000],
            "muR": [4000],
            "mdR": [4000],
        }

        new_points_float = new_points.tolist()
        new_points_rounded = [[round(point, 2) for point in points] for points in new_points_float]

        # Delete first entry for selected_params
        for key in selected_params:
            if parameters[key]:  
                parameters[key].pop(0)  

        # Extend selected parameters by new points
        for index, key in enumerate(selected_params):
            parameters[key].extend(point[index] for point in new_points_rounded)

        new_length = len(parameters["M_1"])

        # Replicate unselected parameters
        for key, value in parameters.items():
            if key not in selected_params:
                parameters[key].extend([value[-1]] * (new_length - len(value)))

    # Output dictionary
    data = {
        "prior": prior_type,
        "num_models": new_length,
        "isGMSB": False,
        "parameters": parameters,
        "steps": [
            {"name": "prep_input", "output_dir": "input", "prefix": "IN"},
            {"name": "SPheno", "input_dir": "input", "output_dir": "SPheno", "log_dir": "SPheno_log", "prefix": "SP"},
            {"name": "softsusy", "input_dir": "input", "output_dir": "softsusy", "prefix": "SS"},
            {"name": "micromegas", "input_dir": "SPheno", "output_dir": "micromegas", "prefix": "MO"},
            {"name": "superiso", "input_dir": "SPheno", "output_dir": "superiso", "prefix": "SI"},
            {"name": "gm2calc", "input_dir": "SPheno", "output_dir": "gm2calc", "prefix": "GM2"},
            {"name": "evade", "input_dir": "SPheno", "output_dir": "evade", "prefix": "EV"}
        ]
    }

    USER = os.environ.get("USER")
    output_dir = f'/u/{USER}/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/data'
    os.makedirs(output_dir, exist_ok=True)

    dumper = InlineListDumper
    dumper.add_representer(list, dumper.represent_list)

    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False, Dumper=dumper)

    print(f"[INFO] YAML file has been created: {output_path}")
