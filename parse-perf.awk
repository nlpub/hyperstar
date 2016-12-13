#!/usr/bin/awk -f
BEGIN {
    OFS = "\t";
    print "clusters", "seconds";
}
match($0, /done in (.+?)\.$/) {
    len  = split(substr($0, RSTART + 8, RLENGTH - 9), time, ":");
    for (i = len; i > 0; i--) seconds += (time[i] + 0.) * 60 ^ (len - i);
    clusters++;
}
END {
    print clusters, seconds;
}
