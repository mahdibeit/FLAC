set root=C:\Users\john.doe\AppData\Local\Continuum\anaconda3

call %root%\Scripts\activate.bat %root%
#!/bin/bash

#python server.py &
#sleep 3 # Sleep for 2s to give the server enough time to start


echo "Press 1 for CIFAR-10 Dataset"

echo "Press 2 for Speech Dataset"

echo "Press 3 for MNIST Dataset"

echo

read -p "Enter the number for dataset: " num
case "$num" in
    1)
    echo "CIFAR-10 Dataset"
     for i in `seq 0 9`; do
         echo "Starting client $i"
         python Clients/Client-CIFAR.py --seed=$i &
     done;;
    2)
    echo "Speech Dataset"
    for i in `seq 0 9`; do
        echo "Starting client $i"
        python Clients/Client-MNIST.py --seed=$i &
    done;;
    3)
    echo "MNIST Dataset"
    for i in `seq 0 9`; do
        echo "Starting client $i"
        python Clients/Client-SpeechCommands.py --seed=$i &
    done;;
    
     
    *) echo "Enter a correct number" ;;
esac










# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

trap 'sleep infinity' EXIT
