import sys
import subprocess

from data.nuim_dataloader import ICLNUIMDataset

if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise ValueError('Usage: ' + sys.argv[0] + ' <path_to_icl_with_camera_angles> <path_to_povray_icl_folder>')

    d = ICLNUIMDataset(sys.argv[1])

    for i in range(d.__len__()):
        item = d.__getitem__(i)
        RT = item['cam']['RT1']
        RTinv = item['cam']['RT1inv']
        print(RT)

        bashCommand = f'povray +Iliving_room.pov +Oscene_00_{i:04d}.png +W640 +H480 ' \
                      f'Declare=val00={RT[0,0]} Declare=val01={RT[1,0]} Declare=val02={RT[2,0]} ' \
                      f'Declare=val10={RT[0,1]} Declare=val11={RT[1,1]} Declare=val12={RT[2,1]} ' \
                      f'Declare=val20={RT[0,2]} Declare=val21={RT[1,2]} Declare=val22={RT[2,2]} ' \
                      f'Declare=val30={RT[0,3]}  Declare=val31={RT[1,3]} Declare=val32={RT[2,3]} ' \
                      f'+FN16 +wt1 -d +L/usr/share/povray-3.7/include Declare=use_baking=2 +A0.0'

        print(bashCommand)

        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=sys.argv[2])
        output, error = process.communicate()

        print(output)
        print(error)