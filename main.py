import numpy as np
from scipy.signal import savgol_filter

common_wave = np.arange(100, 3000, 2)

def read_oneFile(fileName: str, need_list: bool = False):
    filt = set()
    waves = []
    intensitys = []
    result = []
    
    with open(fileName, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "#X" in line:
                continue
            
            parts = line.split()
            if len(parts) != 4:
                print(f"Пропущена строка (не 4 числа): {line}")
                continue
                
            x, y, wave, intensity = map(float, parts)
            
            if (x, y) in filt:
                waves[-1].append(wave)
                intensitys[-1].append(intensity)
            else:
                if waves:
                    sorted_pairs = sorted(zip(waves[-1], intensitys[-1]))
                    xp = [p[0] for p in sorted_pairs]
                    fp = [p[1] for p in sorted_pairs]
                    interp_vals = np.interp(common_wave, xp, fp)
                    result.append(interp_vals)

                filt.add((x, y))
                waves.append([wave])
                intensitys.append([intensity])
    
    if waves:
        sorted_pairs = sorted(zip(waves[-1], intensitys[-1]))
        xp = [p[0] for p in sorted_pairs]
        fp = [p[1] for p in sorted_pairs]
        interp_vals = np.interp(common_wave, xp, fp)
        result.append(interp_vals)

    result = np.array(result)
    result = savgol_filter(result, window_length=11, polyorder=2, axis=1)
    if need_list:
        return result[0]
    return result



if __name__ == "__init__":
    X, y = read_oneFile("control/mk1/prob.txt")