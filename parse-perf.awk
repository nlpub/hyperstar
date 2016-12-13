#!/usr/bin/awk -f
BEGIN {
    OFS = "\t";
    print "cluster", "seconds";
}
match($0, /Cluster ([[:digit:]]+) done in/) {
    cluster = substr($0, RSTART + 8, RLENGTH - 16);
    time    = substr($0, RSTART + RLENGTH, length() - RSTART - RLENGTH);
    len     = split(time, fields, ":");
    for (i = len; i > 0; i--) seconds[cluster] += (fields[i] + 0.) * 60 ^ (len - i);
    count[cluster]++;
}
END {
    for (cluster in seconds) {
        print cluster, seconds[cluster] / count[cluster] | "sort -t'\t' -k1n";
    }
}
