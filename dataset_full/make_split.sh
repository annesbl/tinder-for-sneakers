#!/usr/bin/env bash
shopt -s nullglob
files=(images/*.jpg images/*.jpeg images/*.png)
total=${#files[@]}

train_end=$(( total * 80 / 100 ))
val_end=$(( total * 90 / 100 ))

mkdir -p images/{train,val,test} labels/{train,val,test}

for idx in "${!files[@]}"; do
  img=${files[$idx]}
  lbl=labels/$(basename "${img%.*}.txt")

  if   (( idx < train_end )); then split=train
  elif (( idx < val_end   )); then split=val
  else                             split=test
  fi

  mv -- "$img" "images/$split/"
  mv -- "$lbl" "labels/$split/"
done
