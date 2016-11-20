#!/usr/bin/awk -f
BEGIN {
    OFS = "\t";
    print "directory", "model", "A@1", "A@2", "A@3", "A@4", "A@5", "A@6", "A@7", "A@8", "A@9", "A@10", "AUC";
}
/overall/ {
    match($0, /^For "(.+?)": overall (.+?). AUC=([[:digit:]]+\.[[:digit:]]+).$/, matched);
    match(matched[1], /^(.+)\/(.+?)$/, path);
    len = split(matched[2], ats, ", ");
    for (i = 1; i <= len; i++) { match(ats[i], /[[:digit:]]+\.[[:digit:]]+$/, value); ats[i] = value[0]; }
    auc = matched[3];
    print path[1], path[2], ats[1], ats[2], ats[3], ats[4], ats[5], ats[6], ats[7], ats[8], ats[9], ats[10], auc;
}
