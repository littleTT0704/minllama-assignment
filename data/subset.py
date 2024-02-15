for div in ["train", "test", "dev"]:
    with open(f"sst-{div}.txt", "r", encoding="utf-8") as fin, open(
        f"sst-{div}-subset.txt", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            label, sent = line.split(" ||| ")

            if label in ["0", "1"]:
                new_label = "0"
            elif label in ["3", "4"]:
                new_label = "1"

            if label != "2":
                fout.write(f"{new_label} ||| {sent}")
