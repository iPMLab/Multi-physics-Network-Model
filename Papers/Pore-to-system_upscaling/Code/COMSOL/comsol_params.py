import numpy as np
from pathlib import Path
import re
import h5py

Path_root_Data = Path(
    "D:/yjp/Workdir/Code/ZJU/Study/Python/multi-physic-network-model/Papers/P1/Data"
)


# solid_lambda 400
# void_lambda 0.5942265220905458
def extract_u_hf(filename, _u_astype=str, _hf_astype=str):
    u_match = re.search(r"_u((?:\d+.)*\d+)", filename)
    hf_match = re.search(r"_hf((?:\d+.)*\d+)", filename)
    return (
        _u_astype(u_match.group(1)) if u_match else None,
        _hf_astype(int(hf_match.group(1))) if hf_match else None,
    )


def extract_Re_hf(filename, _Re_astype=str, _hf_astype=str):
    Re_match = re.search(r"_Re((?:\d+.)*\d+)", filename)
    hf_match = re.search(r"_hf((?:\d+.)*\d+)", filename)
    return (
        _Re_astype(Re_match.group(1)) if Re_match else None,
        _hf_astype(int(hf_match.group(1))) if hf_match else None,
    )


def extract_u_hf_fromh5(h5data, _u_astype=str, _hf_astype=str):
    _u_hf_list = []
    for key in h5data.keys():
        _u, _hf = extract_u_hf(key, _u_astype=_u_astype, _hf_astype=_hf_astype)
        _u_hf_list.append([_u, _hf])

    return np.array(_u_hf_list)


def extract_complete_params(input_str, prefix_list):
    escaped_prefixes = [re.escape(prefix) for prefix in prefix_list]
    # 匹配一个或多个 "前缀+数字" 的组合（如 _N20_data1_20）
    pattern = (
        r"(?:"
        + "|".join(escaped_prefixes)
        + r")\d+(?:_\d+)*(?:(?:"
        + "|".join(escaped_prefixes)
        + r")\d+(?:_\d+)*)*"
    )
    matches = re.findall(pattern, input_str)

    # 将匹配结果中的 _ 替换为 .（仅限数字部分）
    processed_matches = []
    for match in matches:
        # 按前缀分割，处理数字部分
        parts = re.split("(" + "|".join(escaped_prefixes) + r")", match)
        processed = []
        for part in parts:
            if part in prefix_list:
                processed.append(part)
            else:
                # 替换数字间的 _ 为 .
                processed.append(part.replace("_", "."))
        processed_matches.append("".join(processed))

    return processed_matches


class ComsolParamsBase:
    # 必须覆盖的静态参数
    _required_attrs = {
        "prefix",
        # "u_inlets",
        "heat_flux_in",
        "heat_flux_out",
        "Path_root",
        # "area_outlet",
        "num_void",
        "num_solid",
        "raw_shape",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 获取当前类和所有父类的属性
        all_attrs = {}
        for base in reversed(cls.__mro__):
            all_attrs.update(
                {k: v for k, v in base.__dict__.items() if not k.startswith("__")}
            )

        # 检查必须覆盖的属性
        missing = cls._required_attrs - set(all_attrs)
        if missing:
            raise AttributeError(
                f"子类 {cls.__name__} 必须显式覆盖以下静态参数: {', '.join(sorted(missing))}"
            )

        # 自动生成路径相关属性
        cls._generate_paths()

    @classmethod
    def _generate_paths(cls):
        """自动生成路径相关属性"""

        cls.Path_comsol = cls.Path_root / "comsol_data"

        cls.Path_pne = cls.Path_root / "pne"

        cls.Path_data_h5 = cls.Path_comsol / "Vdata.h5"
        # cls.Path_mesh_h5 = cls.Path_comsol / f"{cls.prefix}_mesh.h5"

        cls.Path_vtps = cls.Path_pne / "vtps"

        cls.Path_binary_raw = (
            cls.Path_pne
            / f"image_{cls.raw_shape[0]}_{cls.raw_shape[1]}_{cls.raw_shape[2]}.raw"
        )

        cls.Path_mix_raw = (
            cls.Path_pne
            / "images"
            / f"image_{cls.raw_shape[0]}_{cls.raw_shape[1]}_{cls.raw_shape[2]}_mix.raw"
        )

        cls.name_pnm = f"image_{cls.raw_shape[0]}_{cls.raw_shape[1]}_{cls.raw_shape[2]}"

        cls.Path_net_pore = cls.Path_vtps / "pore_network.vtp"

        cls.Path_net_solid = cls.Path_vtps / "solid_network.vtp"

        cls.Path_net_dual = cls.Path_vtps / "dual_network.vtp"

        cls.Path_results = cls.Path_root / "results"

        cls.Path_comsol_xyz = cls.Path_comsol / "xyz.csv"
        cls.Path_PNdata = cls.Path_comsol / "PNdata.pkl"
        cls.Path_interface_coords = cls.Path_comsol / "interface_coords.txt"
        cls.Path_mesh_vtu = cls.Path_comsol / "mesh.vtu"

        # 计算派生属性
        cls.num_pore = cls.num_void + cls.num_solid

        # Material properties
        cls.k_void = 0.5942265220905381
        cls.mu_void = 0.0010093473377638384
        cls.rho_void = 998.2383064921967
        cls.Cp_void = 4186.918113548543

        cls.k_solid = 1.38
        cls.Cp_solid = 703.0
        cls.rho_solid = 2203.0  # kg/m^3
        cls.Res = np.array([0.001, 0.005, 0.02, 0.1, 1])
        cls.Path_figs = cls.Path_results / "figs"

    # 共享的常量
    resolution = 1e-3 / 500
    comsol_params = [
        "p",
        "T",
        "spf.U",
        "u",
        "v",
        "w",
        "ht.cfluxx",
        "ht.cfluxy",
        "ht.cfluxz",
        "ht.dfluxx",
        "ht.dfluxy",
        "ht.dfluxz",
    ]
    comsol_params_map = {k: v for k, v in zip(comsol_params, range(len(comsol_params)))}
    comsol_params_fluid = ["p", "spf.U", "u", "v", "w"]
    num_comsol_params = len(comsol_params_map)


def get_formatted_prefix(input_str, prefix_list=None):
    if prefix_list is None:
        prefix_list = ("_N", "_sample")
    escaped_prefixes = [re.escape(prefix) for prefix in prefix_list]
    # 匹配一个或多个 "前缀+数字" 的组合（如 _N20_data1_20）
    pattern = (
        r"(?:"
        + "|".join(escaped_prefixes)
        + r")\d+(?:_\d+)*(?:(?:"
        + "|".join(escaped_prefixes)
        + r")\d+(?:_\d+)*)*"
    )
    matches = re.findall(pattern, input_str)

    # 将匹配结果中的 _ 替换为 .（仅限数字部分）
    processed_matches = []
    for match in matches:
        # 按前缀分割，处理数字部分
        parts = re.split("(" + "|".join(escaped_prefixes) + r")", match)
        processed = []
        for part in parts:
            if part in prefix_list:
                processed.append(part)
            else:
                # 替换数字间的 _ 为 .
                processed.append(part.replace("_", "."))
        processed_matches.append("".join(processed))

    return processed_matches[0] if len(processed_matches) == 1 else processed_matches


class ComsolParams_N2_500(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (125, 125, 125)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N2_500_sample0(ComsolParams_N2_500):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 37
    num_solid = 49
    area_inlet = 1.7239319502022434e-8
    heat_flux_out = np.array(
        [
            1.4321170775060524e-6,
            7.114297377934893e-6,
            2.7801076246057574e-5,
            1.2310594525120452e-4,
            4.59638633405087e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.20429206526562174,
            1.0214626878724022,
            4.085926055336925,
            20.43030358067441,
            204.4091822988732,
        ]
    )


class ComsolParams_N2_500_sample1(ComsolParams_N2_500):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 42
    num_solid = 52
    area_inlet = 2.157069959047765e-8
    heat_flux_out = np.array(
        [
            1.4540908499587603e-6,
            7.225020413008008e-6,
            2.8261569899031584e-5,
            1.2533677696927097e-4,
            4.641425116546269e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.21278367782839433,
            1.0639250890438499,
            4.256051632057001,
            21.281742743018714,
            213.18271047652007,
        ]
    )


class ComsolParams_N2_500_sample2(ComsolParams_N2_500):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 52
    num_solid = 54
    area_inlet = 1.855976414648925e-8
    heat_flux_out = np.array(
        [
            1.3951822855535574e-6,
            6.941160424588941e-6,
            2.726606365632284e-5,
            1.235927036486081e-4,
            4.813853943050824e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.22188446347064847,
            1.1094300508996702,
            4.437854661677227,
            22.191589620844134,
            222.22536146682774,
        ]
    )


class ComsolParams_N2_500_sample3(ComsolParams_N2_500):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 38
    num_solid = 52
    area_inlet = 2.1797028242249215e-8
    heat_flux_out = np.array(
        [
            1.4641124671467268e-6,
            7.272955603271667e-6,
            2.842587660989408e-5,
            1.259485219060064e-4,
            4.6806957617193714e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.1944181265980946,
            0.9721048371058919,
            3.888862172908074,
            19.383884438665053,
            194.79535698007712,
        ]
    )


class ComsolParams_N2_500_sample4(ComsolParams_N2_500):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 44
    num_solid = 50
    area_inlet = 1.942084262392232e-8
    heat_flux_out = np.array(
        [
            1.4175780609746362e-6,
            7.048331894975711e-6,
            2.76378261229328e-5,
            1.2423475440348722e-4,
            4.710290779728028e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.19448489399356517,
            0.9724318193906359,
            3.8900137592259774,
            19.453338532830074,
            194.92799287021205,
        ]
    )


class ComsolParams_N3_455(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    print()
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (173, 173, 173)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N3_455_sample0(ComsolParams_N3_455):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 92
    num_solid = 106
    area_inlet = 3.7301671689053325e-8
    heat_flux_out = np.array(
        [
            2.702182750138883e-6,
            1.3430096161615022e-5,
            5.254651632428072e-5,
            2.341802835583567e-4,
            8.89944338004945e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.1694766812544241,
            0.8473877226006442,
            3.38961899533692,
            16.9497210929647,
            169.6901316453263,
        ]
    )


class ComsolParams_N3_455_sample1(ComsolParams_N3_455):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 98
    num_solid = 127
    area_inlet = 4.38003935418043e-8
    heat_flux_out = np.array(
        [
            2.744142513439453e-6,
            1.3639437831412272e-5,
            5.341955345680313e-5,
            2.3893738725692997e-4,
            9.188277376292228e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.17157495132269254,
            0.8578776369017129,
            3.4317658737405474,
            17.16006672705532,
            171.7611315850214,
        ]
    )


class ComsolParams_N3_455_sample2(ComsolParams_N3_455):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 93
    num_solid = 111
    area_inlet = 3.747989085614576e-8
    heat_flux_out = np.array(
        [
            2.635781317683224e-6,
            1.3100346088926438e-5,
            5.129792118270356e-5,
            2.293887598687275e-4,
            8.977677811368677e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.19374979970021902,
            0.9687496555935027,
            3.875239938305257,
            19.37656164013901,
            193.82946281651897,
        ]
    )


class ComsolParams_N3_455_sample3(ComsolParams_N3_455):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 104
    num_solid = 116
    area_inlet = 4.106405908519807e-8
    heat_flux_out = np.array(
        [
            2.602213022137246e-6,
            1.2937868009573096e-5,
            5.072078741021577e-5,
            2.279990195742038e-4,
            9.024606673209713e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.18811597237376454,
            0.9405828080825893,
            3.762608938066768,
            18.81430493906189,
            188.30021803336686,
        ]
    )


class ComsolParams_N3_455_sample4(ComsolParams_N3_455):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 108
    num_solid = 117
    area_inlet = 3.998131479009449e-8
    heat_flux_out = np.array(
        [
            2.677048027315991e-6,
            1.3306125966388422e-5,
            5.212751892023811e-5,
            2.3347158253316782e-4,
            9.041170623961611e-4,
        ]
    )
    pressure_out = np.array(
        [
            0.21123667208291833,
            1.0561839614181767,
            4.225000569201305,
            21.125355936934476,
            211.32048482875342,
        ]
    )


class ComsolParams_N4_353(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (218, 218, 218)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N4_353_sample0(ComsolParams_N4_353):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 177
    num_solid = 184
    area_inlet = 6.220335210232041e-8
    heat_flux_out = np.array(
        [
            4.175818555850906e-6,
            2.0319457345185608e-5,
            7.886548762274548e-5,
            3.5627145874012555e-4,
            0.0014142466178656008,
        ]
    )
    pressure_out = np.array(
        [
            0.16991141278923483,
            0.8495596039783457,
            3.3982800101196227,
            16.991414446446097,
            170.09745319122226,
        ]
    )


class ComsolParams_N4_353_sample1(ComsolParams_N4_353):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 189
    num_solid = 208
    area_inlet = 6.454488555333323e-8
    heat_flux_out = np.array(
        [
            4.27454937784278e-6,
            2.081642865509761e-5,
            8.082369939836815e-5,
            3.659366094666411e-4,
            0.0014587679142328736,
        ]
    )
    pressure_out = np.array(
        [
            0.17382482033134605,
            0.8691289816868513,
            3.476552866994833,
            17.382533593478133,
            173.9784264575026,
        ]
    )


class ComsolParams_N4_353_sample2(ComsolParams_N4_353):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 172
    num_solid = 204
    area_inlet = 6.019065501483253e-8
    heat_flux_out = np.array(
        [
            4.076254034304773e-6,
            1.977970797914109e-5,
            7.688301103093418e-5,
            3.483874442848941e-4,
            0.0014062094656932462,
        ]
    )
    pressure_out = np.array(
        [
            0.17859240221245193,
            0.8929635064122472,
            3.5718970866118025,
            17.85958095245213,
            178.7582335131854,
        ]
    )


class ComsolParams_N4_353_sample3(ComsolParams_N4_353):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 185
    num_solid = 200
    area_inlet = 6.202154121377415e-8
    heat_flux_out = np.array(
        [
            4.133276597368902e-6,
            2.0035363422592977e-5,
            7.908286751714834e-5,
            3.5194596342347456e-4,
            0.001440737420684258,
        ]
    )
    pressure_out = np.array(
        [
            0.19268060886698055,
            0.9634007345358581,
            3.853521291769389,
            19.245963808842784,
            192.49628066932496,
        ]
    )


class ComsolParams_N4_353_sample4(ComsolParams_N4_353):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 180
    num_solid = 218
    area_inlet = 6.536376425766311e-8
    heat_flux_out = np.array(
        [
            4.264547294892424e-6,
            2.0709482489439262e-5,
            8.025039443331525e-5,
            3.6200370001289055e-4,
            0.0014178432165581668,
        ]
    )
    pressure_out = np.array(
        [
            0.19222486897403882,
            0.9612876961979374,
            3.845238598029859,
            19.227350654833785,
            192.58699937316203,
        ]
    )


class ComsolParams_N4_689(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (234, 234, 234)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N4_689_sample0(ComsolParams_N4_689):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 214
    num_solid = 257
    area_inlet = 7.73605665185313e-8
    heat_flux_out = np.array(
        [
            4.861215558149942e-6,
            2.3814941216507344e-5,
            9.236276070848815e-5,
            4.1454586649822956e-4,
            0.0016235389500529435,
        ]
    )
    pressure_out = np.array(
        [
            0.1723707487606475,
            0.8618546693499267,
            3.4474457507937872,
            17.236726039975828,
            172.48106951811488,
        ]
    )


class ComsolParams_N4_689_sample1(ComsolParams_N4_689):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 230
    num_solid = 247
    area_inlet = 7.51335170822478e-8
    heat_flux_out = np.array(
        [
            4.882160805708167e-6,
            2.3885238161902874e-5,
            9.446476276658506e-5,
            4.194926431573692e-4,
            0.0016735798078031004,
        ]
    )
    pressure_out = np.array(
        [
            0.16768695755004034,
            0.8384371796020911,
            3.3537264998245755,
            16.769114989646244,
            167.86910892806438,
        ]
    )


class ComsolParams_N4_689_sample2(ComsolParams_N4_689):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 208
    num_solid = 252
    area_inlet = 7.02945391347718e-8
    heat_flux_out = np.array(
        [
            4.629402136117519e-6,
            2.2674064343269384e-5,
            8.967533429183519e-5,
            4.0041962113968304e-4,
            0.0016812383779611295,
        ]
    )
    pressure_out = np.array(
        [
            0.17247028029764283,
            0.86235201981236,
            3.449360274462895,
            17.2466467991825,
            172.57205710195606,
        ]
    )


class ComsolParams_N4_689_sample3(ComsolParams_N4_689):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 231
    num_solid = 242
    area_inlet = 7.185865943439125e-8
    heat_flux_out = np.array(
        [
            4.818482214779296e-6,
            2.3704308357379848e-5,
            9.413419135496724e-5,
            4.165751570189752e-4,
            0.0017001653722576806,
        ]
    )
    pressure_out = np.array(
        [
            0.20008963513241587,
            1.0004496518937598,
            4.0011882340538625,
            20.008689505421685,
            200.2128611799459,
        ]
    )


class ComsolParams_N4_689_sample4(ComsolParams_N4_689):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 215
    num_solid = 253
    area_inlet = 7.677688279881265e-8
    heat_flux_out = np.array(
        [
            4.7280118572993165e-6,
            2.3078826872142193e-5,
            9.10243316366205e-5,
            4.04656559611137e-4,
            0.0016254948721895295,
        ]
    )
    pressure_out = np.array(
        [
            0.17370810127053074,
            0.8685645334399926,
            3.4742226199355875,
            17.371298119034993,
            173.85266870185734,
        ]
    )


class ComsolParams_N4_869(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (243, 243, 243)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N4_869_sample0(ComsolParams_N4_869):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 243
    num_solid = 274
    area_inlet = 8.52777122600362e-8
    heat_flux_out = np.array(
        [
            5.242541418895475e-6,
            2.560116969531135e-5,
            1.0130939077419164e-4,
            4.4697479079548464e-4,
            0.0017592959368925875,
        ]
    )
    pressure_out = np.array(
        [
            0.1756446435034858,
            0.87822342827067,
            3.512809091591858,
            17.564010320909393,
            175.72921069194405,
        ]
    )


class ComsolParams_N4_869_sample1(ComsolParams_N4_869):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 237
    num_solid = 297
    area_inlet = 7.937416581987292e-8
    heat_flux_out = np.array(
        [
            5.262156579958149e-6,
            2.5773453009291817e-5,
            1.0225853238620012e-4,
            4.479925080232155e-4,
            0.001801132201332963,
        ]
    )
    pressure_out = np.array(
        [
            0.1673895172001287,
            0.8369492911077265,
            3.347743460787302,
            16.73557286822605,
            167.52009556748763,
        ]
    )


class ComsolParams_N4_869_sample2(ComsolParams_N4_869):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 284
    area_inlet = 7.429767763772614e-8
    heat_flux_out = np.array(
        [
            5.095421285846215e-6,
            2.459797021285674e-5,
            9.742088009050515e-5,
            4.301087583016216e-4,
            0.0017715486967155023,
        ]
    )
    pressure_out = np.array(
        [
            0.18412597086324117,
            0.9206499291276786,
            3.682535808695476,
            18.412338032564016,
            184.2264118588304,
        ]
    )


class ComsolParams_N4_869_sample3(ComsolParams_N4_869):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 270
    num_solid = 263
    area_inlet = 7.734255238775385e-8
    heat_flux_out = np.array(
        [
            5.191140247713536e-6,
            2.551816773262163e-5,
            1.0109485257486393e-4,
            4.505566674532597e-4,
            0.0018429839185675498,
        ]
    )
    pressure_out = np.array(
        [
            0.2053777449049381,
            1.0268911977649011,
            4.107465624743468,
            20.53729534044621,
            205.48292005280445,
        ]
    )


class ComsolParams_N4_869_sample4(ComsolParams_N4_869):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 239
    num_solid = 282
    area_inlet = 8.3673640464019e-8
    heat_flux_out = np.array(
        [
            4.941112695374589e-6,
            2.4393524504224204e-5,
            9.644007525055187e-5,
            4.27996339896905e-4,
            0.0017460775202376097,
        ]
    )
    pressure_out = np.array(
        [
            0.17031761910593265,
            0.8515896410758385,
            3.406318154175771,
            17.031832901401845,
            170.455816623132,
        ]
    )


class ComsolParams_N5_000(ComsolParamsBase):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (250, 250, 250)

    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N5_000_sample0(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 262
    num_solid = 288

    area_inlet = 8.980615111747199e-8  # 唯一需要覆盖的属性
    heat_flux_out = np.array(
        [
            5.497789811039782e-6,
            2.694968088579419e-5,
            1.0692663326487691e-4,
            4.713940535271018e-4,
            0.0018540977625642859,
        ]
    )
    pressure_out = np.array(
        [
            0.17963053113056426,
            0.8981536739737346,
            3.592410147772337,
            17.96266860643684,
            179.71912389220236,
        ]
    )


class ComsolParams_N5_000_sample1(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 303
    area_inlet = 8.234292049375452e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.428842001409016e-6,
            2.659771522961009e-5,
            1.0516409804824618e-4,
            4.672282251163454e-4,
            0.0019079243147843061,
        ]
    )
    pressure_out = np.array(
        [
            0.16503521858006573,
            0.8251773344387322,
            3.300262644776228,
            16.5024619898934,
            165.0997471976384,
        ]
    )


class ComsolParams_N5_000_sample2(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 285
    num_solid = 293
    area_inlet = 7.853144703226101e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.285177353432845e-6,
            2.591952079088348e-5,
            1.0256246950036416e-4,
            4.5068886293415526e-4,
            0.0018773302786909518,
        ]
    )

    pressure_out = np.array(
        [
            0.18408749040338418,
            0.9204359808011716,
            3.681686388976183,
            18.40708727836465,
            184.2169104353813,
        ]
    )


class ComsolParams_N5_000_sample3(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 281
    num_solid = 282
    area_inlet = 8.297930073114706e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.475173421728435e-6,
            2.6767721258453075e-5,
            1.062786793939534e-4,
            4.6662674077183855e-4,
            0.0019510299589367687,
        ]
    )
    pressure_out = np.array(
        [
            0.20539253884749792,
            1.026960677869797,
            4.107728606905002,
            20.53638211493734,
            205.47971750051133,
        ]
    )


class ComsolParams_N5_000_sample4(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 253
    num_solid = 298
    area_inlet = 8.816164296875864e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.182363078898921e-6,
            2.5538044843064488e-5,
            1.0094259423962331e-4,
            4.4890979261833874e-4,
            0.0018415005618793363,
        ]
    )

    pressure_out = np.array(
        [
            0.17128096946612362,
            0.8564070021744818,
            3.4256585804246784,
            17.128961778343854,
            171.39978472547625,
        ]
    )


class ComsolParams_N5_000_sample20(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 262
    num_solid = 288

    area_inlet = 8.980615111747199e-8  # 唯一需要覆盖的属性
    heat_flux_out = np.array(
        [
            5.497789811039782e-6,
            2.694968088579419e-5,
            1.0692663326487691e-4,
            4.713940535271018e-4,
            0.0018540977625642859,
        ]
    )
    pressure_out = np.array(
        [
            0.17963053113056426,
            0.8981536739737346,
            3.592410147772337,
            17.96266860643684,
            179.71912389220236,
        ]
    )


class ComsolParams_N5_000_sample21(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 303
    area_inlet = 8.234292049375452e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.428842001409016e-6,
            2.659771522961009e-5,
            1.0516409804824618e-4,
            4.672282251163454e-4,
            0.0019079243147843061,
        ]
    )
    pressure_out = np.array(
        [
            0.16503521858006573,
            0.8251773344387322,
            3.300262644776228,
            16.5024619898934,
            165.0997471976384,
        ]
    )


class ComsolParams_N5_000_sample22(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 285
    num_solid = 293
    area_inlet = 7.853144703226101e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.285177353432845e-6,
            2.591952079088348e-5,
            1.0256246950036416e-4,
            4.5068886293415526e-4,
            0.0018773302786909518,
        ]
    )

    pressure_out = np.array(
        [
            0.18408749040338418,
            0.9204359808011716,
            3.681686388976183,
            18.40708727836465,
            184.2169104353813,
        ]
    )


class ComsolParams_N5_000_sample23(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 281
    num_solid = 282
    area_inlet = 8.297930073114706e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.475173421728435e-6,
            2.6767721258453075e-5,
            1.062786793939534e-4,
            4.6662674077183855e-4,
            0.0019510299589367687,
        ]
    )
    pressure_out = np.array(
        [
            0.20539253884749792,
            1.026960677869797,
            4.107728606905002,
            20.53638211493734,
            205.47971750051133,
        ]
    )


class ComsolParams_N5_000_sample24(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 253
    num_solid = 298
    area_inlet = 8.816164296875864e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.182363078898921e-6,
            2.5538044843064488e-5,
            1.0094259423962331e-4,
            4.4890979261833874e-4,
            0.0018415005618793363,
        ]
    )

    pressure_out = np.array(
        [
            0.17128096946612362,
            0.8564070021744818,
            3.4256585804246784,
            17.128961778343854,
            171.39978472547625,
        ]
    )


class ComsolParams_N5_000_sample25(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 262
    num_solid = 288

    area_inlet = 8.980615111747199e-8  # 唯一需要覆盖的属性
    heat_flux_out = np.array(
        [
            5.497789811039782e-6,
            2.694968088579419e-5,
            1.0692663326487691e-4,
            4.713940535271018e-4,
            0.0018540977625642859,
        ]
    )
    pressure_out = np.array(
        [
            0.17963053113056426,
            0.8981536739737346,
            3.592410147772337,
            17.96266860643684,
            179.71912389220236,
        ]
    )


class ComsolParams_N5_000_sample26(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 303
    area_inlet = 8.234292049375452e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.428842001409016e-6,
            2.659771522961009e-5,
            1.0516409804824618e-4,
            4.672282251163454e-4,
            0.0019079243147843061,
        ]
    )
    pressure_out = np.array(
        [
            0.16503521858006573,
            0.8251773344387322,
            3.300262644776228,
            16.5024619898934,
            165.0997471976384,
        ]
    )


class ComsolParams_N5_000_sample27(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 285
    num_solid = 293
    area_inlet = 7.853144703226101e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.285177353432845e-6,
            2.591952079088348e-5,
            1.0256246950036416e-4,
            4.5068886293415526e-4,
            0.0018773302786909518,
        ]
    )

    pressure_out = np.array(
        [
            0.18408749040338418,
            0.9204359808011716,
            3.681686388976183,
            18.40708727836465,
            184.2169104353813,
        ]
    )


class ComsolParams_N5_000_sample28(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 281
    num_solid = 282
    area_inlet = 8.297930073114706e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.475173421728435e-6,
            2.6767721258453075e-5,
            1.062786793939534e-4,
            4.6662674077183855e-4,
            0.0019510299589367687,
        ]
    )
    pressure_out = np.array(
        [
            0.20539253884749792,
            1.026960677869797,
            4.107728606905002,
            20.53638211493734,
            205.47971750051133,
        ]
    )


class ComsolParams_N5_000_sample29(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 253
    num_solid = 298
    area_inlet = 8.816164296875864e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.182363078898921e-6,
            2.5538044843064488e-5,
            1.0094259423962331e-4,
            4.4890979261833874e-4,
            0.0018415005618793363,
        ]
    )

    pressure_out = np.array(
        [
            0.17128096946612362,
            0.8564070021744818,
            3.4256585804246784,
            17.128961778343854,
            171.39978472547625,
        ]
    )


class ComsolParams_N5_000_sample30(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 262
    num_solid = 288

    area_inlet = 8.980615111747199e-8  # 唯一需要覆盖的属性
    heat_flux_out = np.array(
        [
            5.497789811039782e-6,
            2.694968088579419e-5,
            1.0692663326487691e-4,
            4.713940535271018e-4,
            0.0018540977625642859,
        ]
    )
    pressure_out = np.array(
        [
            0.17963053113056426,
            0.8981536739737346,
            3.592410147772337,
            17.96266860643684,
            179.71912389220236,
        ]
    )


class ComsolParams_N5_000_sample31(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 303
    area_inlet = 8.234292049375452e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.428842001409016e-6,
            2.659771522961009e-5,
            1.0516409804824618e-4,
            4.672282251163454e-4,
            0.0019079243147843061,
        ]
    )
    pressure_out = np.array(
        [
            0.16503521858006573,
            0.8251773344387322,
            3.300262644776228,
            16.5024619898934,
            165.0997471976384,
        ]
    )


class ComsolParams_N5_000_sample32(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 285
    num_solid = 293
    area_inlet = 7.853144703226101e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.285177353432845e-6,
            2.591952079088348e-5,
            1.0256246950036416e-4,
            4.5068886293415526e-4,
            0.0018773302786909518,
        ]
    )

    pressure_out = np.array(
        [
            0.18408749040338418,
            0.9204359808011716,
            3.681686388976183,
            18.40708727836465,
            184.2169104353813,
        ]
    )


class ComsolParams_N5_000_sample33(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 281
    num_solid = 282
    area_inlet = 8.297930073114706e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.475173421728435e-6,
            2.6767721258453075e-5,
            1.062786793939534e-4,
            4.6662674077183855e-4,
            0.0019510299589367687,
        ]
    )
    pressure_out = np.array(
        [
            0.20539253884749792,
            1.026960677869797,
            4.107728606905002,
            20.53638211493734,
            205.47971750051133,
        ]
    )


class ComsolParams_N5_000_sample34(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 253
    num_solid = 298
    area_inlet = 8.816164296875864e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.182363078898921e-6,
            2.5538044843064488e-5,
            1.0094259423962331e-4,
            4.4890979261833874e-4,
            0.0018415005618793363,
        ]
    )

    pressure_out = np.array(
        [
            0.17128096946612362,
            0.8564070021744818,
            3.4256585804246784,
            17.128961778343854,
            171.39978472547625,
        ]
    )


class ComsolParams_N5_000_sample35(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 262
    num_solid = 288

    area_inlet = 8.980615111747199e-8  # 唯一需要覆盖的属性
    heat_flux_out = np.array(
        [
            5.497789811039782e-6,
            2.694968088579419e-5,
            1.0692663326487691e-4,
            4.713940535271018e-4,
            0.0018540977625642859,
        ]
    )
    pressure_out = np.array(
        [
            0.17963053113056426,
            0.8981536739737346,
            3.592410147772337,
            17.96266860643684,
            179.71912389220236,
        ]
    )


class ComsolParams_N5_000_sample36(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 251
    num_solid = 303
    area_inlet = 8.234292049375452e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.428842001409016e-6,
            2.659771522961009e-5,
            1.0516409804824618e-4,
            4.672282251163454e-4,
            0.0019079243147843061,
        ]
    )
    pressure_out = np.array(
        [
            0.16503521858006573,
            0.8251773344387322,
            3.300262644776228,
            16.5024619898934,
            165.0997471976384,
        ]
    )


class ComsolParams_N5_000_sample37(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 285
    num_solid = 293
    area_inlet = 7.853144703226101e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.285177353432845e-6,
            2.591952079088348e-5,
            1.0256246950036416e-4,
            4.5068886293415526e-4,
            0.0018773302786909518,
        ]
    )

    pressure_out = np.array(
        [
            0.18408749040338418,
            0.9204359808011716,
            3.681686388976183,
            18.40708727836465,
            184.2169104353813,
        ]
    )


class ComsolParams_N5_000_sample38(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 281
    num_solid = 282
    area_inlet = 8.297930073114706e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.475173421728435e-6,
            2.6767721258453075e-5,
            1.062786793939534e-4,
            4.6662674077183855e-4,
            0.0019510299589367687,
        ]
    )
    pressure_out = np.array(
        [
            0.20539253884749792,
            1.026960677869797,
            4.107728606905002,
            20.53638211493734,
            205.47971750051133,
        ]
    )


class ComsolParams_N5_000_sample39(ComsolParams_N5_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 253
    num_solid = 298
    area_inlet = 8.816164296875864e-8  # 唯一需要覆盖的属
    heat_flux_out = np.array(
        [
            5.182363078898921e-6,
            2.5538044843064488e-5,
            1.0094259423962331e-4,
            4.4890979261833874e-4,
            0.0018415005618793363,
        ]
    )

    pressure_out = np.array(
        [
            0.17128096946612362,
            0.8564070021744818,
            3.4256585804246784,
            17.128961778343854,
            171.39978472547625,
        ]
    )


class ComsolParams_N10_000(ComsolParamsBase):
    """1_16系列的共同参数(中间类)"""

    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 0
    num_solid = 0
    raw_shape = (500, 500, 500)
    heat_flux_in = 100000.0
    heat_flux_out = np.array([0])
    area_inlet = 0


class ComsolParams_N10_000_sample0(ComsolParams_N10_000):
    prefix = get_formatted_prefix(__qualname__)
    Path_root = Path_root_Data / prefix
    num_void = 1840
    num_solid = 2008
    area_inlet = 3.7593297141976103e-7
    heat_flux_out = np.array(
        [
            2.073760647808807e-5,
            1.0237365741834985e-4,
            4.0189642328841685e-4,
            0.001803829899915112,
            0.007496818520215931,
        ]
    )
    pressure_out = np.array(
        [
            0.1743768348048963,
            0.8720091480007288,
            3.4880688145929715,
            17.43771729729628,
            174.44991451729396,
        ]
    )


# print(ComsolParams_N10_000_sample0.prefix)
PARAMS_N2_500 = [
    # ComsolParams_N2_500_sample0,
    # ComsolParams_N2_500_sample1,
    # ComsolParams_N2_500_sample2,
    ComsolParams_N2_500_sample3,
    # ComsolParams_N2_500_sample4,
]
PARAMS_N3_455 = [
    # ComsolParams_N3_455_sample0,
    # ComsolParams_N3_455_sample1,
    # ComsolParams_N3_455_sample2,
    # ComsolParams_N3_455_sample3,
    ComsolParams_N3_455_sample4,
]
PARAMS_N4_353 = [
    # ComsolParams_N4_353_sample0,
    # ComsolParams_N4_353_sample1,
    # ComsolParams_N4_353_sample2,
    ComsolParams_N4_353_sample3,
    # ComsolParams_N4_353_sample4,
]

PARAMS_N4_689 = [
    ComsolParams_N4_689_sample0,
    ComsolParams_N4_689_sample1,
    ComsolParams_N4_689_sample2,
    ComsolParams_N4_689_sample3,
    ComsolParams_N4_689_sample4,
]

PARAMS_N4_869 = [
    ComsolParams_N4_869_sample0,
    # ComsolParams_N4_869_sample1,
    # ComsolParams_N4_869_sample2,
    # ComsolParams_N4_869_sample3,
    # ComsolParams_N4_869_sample4,
]


PARAMS_N5_000 = [
    ComsolParams_N5_000_sample0,
    # ComsolParams_N5_000_sample1,
    # ComsolParams_N5_000_sample2,
    # ComsolParams_N5_000_sample3,
    # ComsolParams_N5_000_sample4,
]

PARAMS_N10_000 = [
    ComsolParams_N10_000_sample0,
]

PARAMS_N5_000_marching_cube = [
    ComsolParams_N5_000_sample20,
    ComsolParams_N5_000_sample21,
    ComsolParams_N5_000_sample22,
    ComsolParams_N5_000_sample23,
    ComsolParams_N5_000_sample24,
]

PARAMS_N5_000_voxel = [
    ComsolParams_N5_000_sample25,
    ComsolParams_N5_000_sample26,
    ComsolParams_N5_000_sample27,
    ComsolParams_N5_000_sample28,
    ComsolParams_N5_000_sample29,
]
PARAMS_N5_000_constrained_smooth = [
    ComsolParams_N5_000_sample30,
    ComsolParams_N5_000_sample31,
    ComsolParams_N5_000_sample32,
    ComsolParams_N5_000_sample33,
    ComsolParams_N5_000_sample34,
]


PARAMS_N5_000_minkowski = [
    ComsolParams_N5_000_sample35,
    ComsolParams_N5_000_sample36,
    ComsolParams_N5_000_sample37,
    ComsolParams_N5_000_sample38,
    ComsolParams_N5_000_sample39,
]


comsol_params_map = ComsolParamsBase.comsol_params_map
