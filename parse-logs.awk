#!/usr/bin/awk -f
BEGIN {
    OFS = "\t";
    print "directory", "model", "hit@1", "hit@2", "hit@3", "hit@4", "hit@5", "hit@6", "hit@7", "hit@8", "hit@9", "hit@10", "AUC";
}
/overall/ {
    match($0, /^For "(.+?)": overall (.+?). AUC=([[:digit:]]+\.[[:digit:]]+).$/, matched);
    match(matched[1], /^(.+)\/(.+?)$/, path);
    len = split(matched[2], ats, ", ");
    for (i = 1; i <= len; i++) { match(ats[i], /[[:digit:]]+\.[[:digit:]]+$/, value); ats[i] = value[0]; }
    auc = matched[3];
    print path[1], path[2], ats[1], ats[2], ats[3], ats[4], ats[5], ats[6], ats[7], ats[8], ats[9], ats[10], auc;
}
