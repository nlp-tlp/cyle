#!/bin/bash
clear="\033[0m"
while [ "$input" != "4" ] || [ "$input" != "q" ] || [ "$input" != "quit" ]; do
	echo
	echo -e "Which dataset would you like to select?" 
    echo -e "
    \\033[32m1. FB15k
    \\033[32m2. FB15k-237
    \\033[32m3. NELL995
    \\033[31m4. Quit${clear}\n"

	read -p $'Enter your choice  \e[32m[1-3]\e[0m \e[31m[4/q/quit]\e[0m: ' input
    echo
    case $input in
        1)
		CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
			--data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 30 -lr 0.00005 --max_steps 450001 \
			--cpu_num 2 --geo cylinder --valid_steps 30000 -projm "(1600,2)" --save_checkpoint_steps 30000 \
			--print_on_screen --seed 0
		break
			;;
        2)
		CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
			--data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 20 -lr 0.00005 --max_steps 350001 \
			--cpu_num 2 --geo cylinder --valid_steps 30000 -projm "(1600,2)" --save_checkpoint_steps 30000 \
			--print_on_screen --seed 0
		break
			;;
        3)
		CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
			--data_path data/NELL-betae -n 128 -b 512 -d 800 -g 20 -lr 0.0001 --max_steps 350001 \
			--cpu_num 2 --geo cylinder --valid_steps 30000 -projm "(1600,2)" --save_checkpoint_steps 30000 \
			--print_on_screen --seed 0
		break
			;;
        4)
            echo "Quit!"
			exit 0
			;;
        q|quit|exit)
            echo "Quit!"
			exit 0
			;;
        *)
			echo -e "\\033[31mInvalid options. Try again.${clear}"
			;;
    esac
done

