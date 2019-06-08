#!/usr/bin/awk -f
BEGIN {
    OFS = "\t";
    print "directory", "model", "hit@1", "hit@2", "hit@3", "hit@4", "hit@5", "hit@6", "hit@7", "hit@8", "hit@9", "hit@10", "AUC";
}
match($0, /^For "(.+?)": overall/) {
    split(substr($0, RSTART + 5, RLENGTH - 15), path, "/");
}
match($0, /overall (.+?). AUC/) {
    split(substr($0, RSTART + 8), hit, ", ");
    for (i = 1; i <= 10; i++) { gsub("\\. .*$", "", hit[i]); gsub("^(.*?)=", "", hit[i]); }
}
match($0, /AUC=(.+)\.$/) {
    auc = substr($0, RSTART);
    gsub("(^AUC=|\\.$)", "", auc);
    print path[1], path[2], hit[1], hit[2], hit[3], hit[4], hit[5], hit[6], hit[7], hit[8], hit[9], hit[10], auc | "sort -n -t'\t' -k1V -k2";
}
