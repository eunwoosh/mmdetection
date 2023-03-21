while getopts "c:m" flag
do
  case "${flag}" in
    c) cuda_idx=${OPTARG} ;;
    m) mixed_precision="true" ;;
  esac
done

if [ -n "$cuda_idx" ]; then
  export CUDA_VISIBLE_DEVICES=$cuda_idx
fi

random_hash=$(echo $RANDOM | md5sum | head -c 20)

if [ -z "$mixed_precision" ]; then
  python ../tools/train.py model.py --work-dir logs/atss_fullp_voc_${random_hash}
else
  python ../tools/train.py modelfp16.py --work-dir logs/atss_mixed_voc_${random_hash}
fi


# python ../tools/train.py model_atss.py | tee logs/test_${random_hash}.log
# python ../tools/test.py model.py 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth' --eval bbox | tee logs/test_${random_hash}.log