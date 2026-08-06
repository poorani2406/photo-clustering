# How face grouping works

This document explains `app/clustering.py` in plain language — no ML
background assumed. It describes the algorithm as currently implemented;
nothing here changes the code's behavior.

## The input

Every detected face has been turned into a **512-dimensional embedding** —
a list of 512 numbers that describes what that face looks like, produced by
the InsightFace model. Two embeddings that are close together (by distance,
or by the angle between them) belong to similar-looking faces. That's the
entire signal the algorithm has to work with: no names, no metadata, just
"how similar is this face to that one."

## Step 1 — DBSCAN: find the obvious groups first

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) is a clustering algorithm that
groups points sitting close together in space, without needing to be told in
advance how many groups to expect. That fits this problem well — the app has
no idea how many people are in a folder of photos until it looks.

- Two knobs control it: **`CLUSTER_EPS`** (how close two embeddings must be
  to count as "the same person") and **`CLUSTER_MIN_SAMPLES`** (how many
  photos are needed before DBSCAN is confident enough to call something a
  cluster on its own).
- Anything DBSCAN can't confidently group — a face that only appears once,
  a blurry photo, an odd side-profile angle — gets left as an "outlier"
  (DBSCAN's label `-1`) rather than being forced into the wrong group.

At this point some real people are correctly identified, and a pile of
leftover faces (outliers, plus any cluster too small to trust) still need a
home.

## Step 2 — deciding which clusters are trustworthy

Not every group DBSCAN finds is treated as reliable. A cluster only becomes
**"established"** if it has at least `CLUSTER_MIN_MERGE_SIZE` faces in it.
Small, shaky clusters are treated the same as outliers — candidates to be
re-evaluated, not answers to build on.

## Step 3 — the iterative reassignment pass

This is the part that's custom to this project, not something DBSCAN does
out of the box.

For every leftover face ("candidate"), the algorithm asks: *does this face
actually belong to one of the established clusters, even though DBSCAN
didn't put it there?* It checks two things, and **both** have to pass:

1. **Centroid similarity** — how similar is this face to the *average*
   (centroid) of everyone already confirmed in that cluster? This is a
   broad "does this look like the general vibe of this person's other
   photos" check.
2. **Face similarity** — how similar is this face to the single
   *closest individual photo* already in that cluster? This is a stricter,
   more literal "does this look like at least one specific photo we're sure
   about" check.

Requiring both prevents two different failure modes: relying only on the
centroid could let a face drift into a cluster because the *average* looks
close enough even if no individual photo really matches; relying only on
individual-face similarity could let one unusual lookalike photo pull in
someone who doesn't belong. Requiring both cuts down on false merges.

If a candidate clears both thresholds against its best-matching cluster, it's
merged into that cluster. If not, it stays a candidate for the next round.

**Why "iterative"?** After every successful merge, the cluster's centroid
shifts slightly (it now includes one more face). So the algorithm rebuilds
each cluster's profile and runs another full pass over the remaining
candidates. A face that didn't quite match a cluster in round 1 might match
it in round 2, once that cluster has absorbed a nearby face and shifted
slightly toward it. This repeats until a full pass produces **zero** new
merges — at which point nothing left is going to change, and the loop stops.

## Step 4 — nobody gets dropped

Any face that never merges into an established cluster still deserves to
show up as *someone* — most commonly, this is a person who only appears once
in the whole folder. Rather than discarding them as noise, each remaining
outlier becomes its own new cluster: a "person" with exactly one photo.

## Step 5 — final labels

Every face now has a final cluster ID. The UI numbers these `Person 1`,
`Person 2`, etc. (cluster ID + 1), and each person's card shows every photo
assigned to their cluster.

## Summary in one paragraph

DBSCAN finds the confident, obvious groups first. Everything it's unsure
about gets a second look: each leftover face is compared against every
confident group both "on average" and "against its single closest match,"
and only merged if it clearly passes both checks. This repeats, since each
merge can make a group a slightly better match for the next leftover face,
until nothing changes. Anyone left over becomes their own one-photo group,
so no detected face is ever silently dropped.
