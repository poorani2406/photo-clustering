"""
Groups every face embedding found across the whole folder into "people".

DBSCAN is a natural fit here because, unlike k-means, it doesn't require
knowing the number of people ahead of time - it just groups points that are
close together in embedding space and leaves outliers unlabeled.

Faces that don't match anyone else closely enough (DBSCAN label -1, e.g. a
person who only appears once, or a blurry/side-profile face) still deserve
to show up as a person of their own, just with one photo - so we give each
of them their own singleton cluster instead of dropping them.
"""
import numpy as np
from sklearn.cluster import DBSCAN

from app.config import (
    CLUSTER_EPS,
    CLUSTER_MIN_SAMPLES,
    CLUSTER_MERGE_THRESHOLD,
    CLUSTER_FACE_THRESHOLD,
    CLUSTER_MIN_MERGE_SIZE,
)


def _build_cluster_profiles(established_labels, current_assignments, ids, embeddings, face_rows):
    cluster_profiles = {}
    for lbl in established_labels:
        assigned_fids = [fid for fid, curr_lbl in current_assignments.items() if curr_lbl == lbl]
        
        lbl_embeddings = []
        lbl_filenames = []
        lbl_face_ids = []
        for fid in assigned_fids:
            row_idx = ids.index(fid)
            emb = embeddings[row_idx]
            norm = np.linalg.norm(emb)
            normed_emb = emb / norm if norm > 0 else emb
            lbl_embeddings.append(normed_emb)
            lbl_filenames.append(face_rows[row_idx].get("filename", "unknown"))
            lbl_face_ids.append(fid)
            
        mean_emb = np.mean(lbl_embeddings, axis=0)
        centroid_norm = np.linalg.norm(mean_emb)
        centroid = mean_emb / centroid_norm if centroid_norm > 0 else mean_emb
        
        cluster_profiles[lbl] = {
            "centroid": centroid,
            "face_embeddings": lbl_embeddings,
            "face_ids": lbl_face_ids,
            "filenames": lbl_filenames,
            "size": len(assigned_fids),
        }
    return cluster_profiles


def cluster_faces(face_rows: list[dict]) -> dict[int, int]:
    """
    face_rows: list of dicts with at least {id, embedding (bytes)}.
    Returns a dict mapping face_id -> cluster_index (0, 1, 2, ...).
    """
    print(f"[DEBUG] [cluster_faces] Received {len(face_rows)} face_rows.")
    if not face_rows:
        print("[DEBUG] [cluster_faces] face_rows is empty. Returning empty assignments dict.")
        return {}

    ids = [f["id"] for f in face_rows]
    embeddings = np.array(
        [np.frombuffer(f["embedding"], dtype=np.float64) for f in face_rows]
    )
    print(f"[DEBUG] [cluster_faces] Formed embeddings array with shape {embeddings.shape}.")

    # Print embedding details for verification
    print("\n=== EMBEDDING DETAILS VERIFICATION ===")
    for face_row, emb in zip(face_rows, embeddings):
        fname = face_row.get("filename", "unknown")
        fid = face_row["id"]
        # Print for Z group files
        if any(fname.startswith(prefix) for prefix in ["z1", "z2", "z3", "z4"]):
            l2_norm = np.linalg.norm(emb)
            print(f"Face ID: {fid}, Filename: {fname}")
            print(f"  dtype: {emb.dtype}")
            print(f"  shape: {emb.shape}")
            print(f"  L2 Norm: {l2_norm:.6f}")
            print(f"  First 5 values: {emb[:5]}")
    print("======================================\n")

    # Step 1: Run DBSCAN normally.
    labels = DBSCAN(
        eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="euclidean"
    ).fit_predict(embeddings)
    print(f"[DEBUG] [cluster_faces] DBSCAN fit_predict completed. Initial labels: {labels}")

    # Count the size of each cluster returned by DBSCAN.
    unique_labels = set(labels)
    label_to_indices = {}
    for lbl in unique_labels:
        label_to_indices[lbl] = [idx for idx, l in enumerate(labels) if l == lbl]

    # Separate established clusters and candidate indices.
    established_labels = set()
    for lbl, indices in label_to_indices.items():
        if lbl != -1 and len(indices) >= CLUSTER_MIN_MERGE_SIZE:
            established_labels.add(lbl)

    candidate_ids = []
    for idx, (face_id, label) in enumerate(zip(ids, labels)):
        if label == -1 or label not in established_labels:
            candidate_ids.append(face_id)

    # We maintain a mapping of face_id to its current cluster label.
    # Initial state is the DBSCAN labels.
    current_assignments = {fid: int(lbl) for fid, lbl in zip(ids, labels)}
    next_cluster_id = labels.max() + 1 if len(labels) else 0

    iteration = 1
    while True:
        print(f"\n=== ITERATIVE REASSIGNMENT: ITERATION {iteration} ===")
        
        # Build initial cluster profiles for this iteration
        cluster_profiles = _build_cluster_profiles(
            established_labels, current_assignments, ids, embeddings, face_rows
        )

        merged_any = False
        merged_in_this_pass = []
        
        # Evaluate every remaining candidate
        for face_id in list(candidate_ids):
            row_idx = ids.index(face_id)
            orig_label = labels[row_idx]
            raw_emb = embeddings[row_idx]
            filename = face_rows[row_idx].get("filename", "unknown")
            
            # Ensure candidate embedding is L2-normalized
            raw_norm = np.linalg.norm(raw_emb)
            outlier_emb = raw_emb / raw_norm if raw_norm > 0 else raw_emb
            
            best_cluster = None
            best_centroid_sim = -1.0
            best_max_face_sim = -1.0
            best_size = 0
            
            print("-----------------------------------")
            print(f"Centroid threshold: {CLUSTER_MERGE_THRESHOLD}")
            print(f"Face threshold: {CLUSTER_FACE_THRESHOLD}\n")
            print(f"Candidate filename: {filename} (Face ID: {face_id})")
            print(f"Current cluster: {current_assignments[face_id] if current_assignments[face_id] != -1 else 'Outlier (-1)'}\n")
            
            for cand_lbl, profile in cluster_profiles.items():
                centroid_sim = float(np.dot(outlier_emb, profile["centroid"]))
                
                print(f"Compare with cluster {cand_lbl} (Person {cand_lbl + 1})")
                print(f"Cluster {cand_lbl} contains:")
                for comp_face_id, comp_filename in zip(profile["face_ids"], profile["filenames"]):
                    print(f"  Face ID: {comp_face_id}, Filename: {comp_filename}")
                    
                print(f"Cluster size:\n{profile['size']}\n")
                print(f"Centroid similarity:\n{centroid_sim:.6f}\n")
                
                print(f"Candidate {filename} comparisons:")
                face_sims = []
                for comp_face_id, comp_filename, face_emb in zip(profile["face_ids"], profile["filenames"], profile["face_embeddings"]):
                    sim = float(np.dot(outlier_emb, face_emb))
                    face_sims.append(sim)
                    print(f"  vs {comp_filename} (Face ID: {comp_face_id}) -> {sim:.6f}")
                    
                max_face_sim = max(face_sims) if face_sims else -1.0
                print(f"Maximum face similarity:\n{max_face_sim:.6f}\n")
                
            passed_clusters = []
            for cand_lbl, profile in cluster_profiles.items():
                centroid_sim = float(np.dot(outlier_emb, profile["centroid"]))
                
                print(f"Compare with cluster {cand_lbl} (Person {cand_lbl + 1})")
                print(f"Cluster {cand_lbl} contains:")
                for comp_face_id, comp_filename in zip(profile["face_ids"], profile["filenames"]):
                    print(f"  Face ID: {comp_face_id}, Filename: {comp_filename}")
                    
                print(f"Cluster size:\n{profile['size']}\n")
                print(f"Centroid similarity:\n{centroid_sim:.6f}\n")
                
                print(f"Candidate {filename} comparisons:")
                face_sims = []
                for comp_face_id, comp_filename, face_emb in zip(profile["face_ids"], profile["filenames"], profile["face_embeddings"]):
                    sim = float(np.dot(outlier_emb, face_emb))
                    face_sims.append(sim)
                    print(f"  vs {comp_filename} (Face ID: {comp_face_id}) -> {sim:.6f}")
                    
                max_face_sim = max(face_sims) if face_sims else -1.0
                print(f"Maximum face similarity:\n{max_face_sim:.6f}\n")
                
                if centroid_sim >= CLUSTER_MERGE_THRESHOLD and max_face_sim >= CLUSTER_FACE_THRESHOLD:
                    passed_clusters.append({
                        "centroid_sim": centroid_sim,
                        "max_face_sim": max_face_sim,
                        "label": cand_lbl,
                        "size": profile["size"]
                    })
                    
            if passed_clusters:
                # Select the best cluster from those that passed both thresholds based on centroid similarity
                passed_clusters.sort(key=lambda x: x["centroid_sim"], reverse=True)
                best_match = passed_clusters[0]
                best_cluster = best_match["label"]
                best_centroid_sim = best_match["centroid_sim"]
                best_max_face_sim = best_match["max_face_sim"]
                merged = True
            else:
                best_cluster = None
                merged = False
            
            if merged:
                current_assignments[face_id] = int(best_cluster)
                candidate_ids.remove(face_id)
                merged_any = True
                merged_in_this_pass.append(face_id)
                
                print(f"Best cluster selected:\n{best_cluster}\n")
                print("Decision:\nMERGED\n")
                print(f"Final cluster assignment:\nPerson {best_cluster + 1}\n")
                
                # OPTIMIZATION: Rebuild cluster profiles immediately after a successful merge
                print("[DEBUG] Successful merge. Rebuilding cluster profiles immediately...")
                cluster_profiles = _build_cluster_profiles(
                    established_labels, current_assignments, ids, embeddings, face_rows
                )
            else:
                print("Decision:\nREJECTED\n")
                print("Reason for rejection:\nNo candidate clusters passed both centroid and face thresholds.\n")
                print(f"Final cluster assignment (temporary):\n{current_assignments[face_id]}\n")
            print("-----------------------------------")
            
        print(f"Merged in iteration {iteration}: {merged_in_this_pass}")
        if not merged_any:
            print(f"No candidates merged in iteration {iteration}. Terminating loop.")
            break
            
        iteration += 1

    # Assign unique final cluster IDs to remaining candidates
    for face_id in candidate_ids:
        orig_lbl = current_assignments[face_id]
        if orig_lbl == -1:
            current_assignments[face_id] = int(next_cluster_id)
            next_cluster_id += 1

    # Hierarchical Merging of Established Clusters (e.g. Person 2 and Person 4)
    current_assignments = _merge_established_clusters(current_assignments, face_rows, embeddings, ids)

    # Add debug output to print mapping for every final person
    final_groups = {}
    for face_row, label, final_lbl in zip(face_rows, labels, [current_assignments[fid] for fid in ids]):
        if final_lbl not in final_groups:
            final_groups[final_lbl] = []
        final_groups[final_lbl].append({
            "face_id": face_row["id"],
            "filename": face_row.get("filename", "unknown"),
            "dbscan_label": label
        })
        
    print("\n=== FINAL CLUSTERING TO UI PERSON MAPPING ===")
    for final_lbl in sorted(final_groups.keys()):
        print(f"Person {final_lbl + 1}")
        print(f"Cluster ID: {final_lbl}")
        # Get unique original DBSCAN labels in this group
        orig_labels = set(item["dbscan_label"] for item in final_groups[final_lbl])
        orig_labels_str = ", ".join(str(l) if l != -1 else "Outlier (-1)" for l in orig_labels)
        print(f"Original DBSCAN label: {orig_labels_str}")
        print("Images:")
        for item in final_groups[final_lbl]:
            print(f"  {item['filename']} (Face ID: {item['face_id']})")
        print()
    print("=============================================\n")

    print(f"[DEBUG] [cluster_faces] Final assignments mapping: {current_assignments}")
    return current_assignments


def _merge_established_clusters(assignments: dict[int, int], face_rows: list[dict], embeddings: np.ndarray, ids: list[int]) -> dict[int, int]:
    """Iteratively merges established clusters whose centroids are highly similar."""
    while True:
        unique_labels = sorted(list(set(assignments.values())))
        if len(unique_labels) < 2:
            break
            
        cluster_profiles = {}
        for lbl in unique_labels:
            assigned_fids = [fid for fid, curr_lbl in assignments.items() if curr_lbl == lbl]
            lbl_embeddings = []
            for fid in assigned_fids:
                row_idx = ids.index(fid)
                emb = embeddings[row_idx]
                norm = np.linalg.norm(emb)
                normed_emb = emb / norm if norm > 0 else emb
                lbl_embeddings.append(normed_emb)
                
            mean_emb = np.mean(lbl_embeddings, axis=0)
            centroid_norm = np.linalg.norm(mean_emb)
            centroid = mean_emb / centroid_norm if centroid_norm > 0 else mean_emb
            cluster_profiles[lbl] = centroid
            
        # Find the best pair of clusters to merge
        best_pair = None
        best_sim = -1.0
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                lbl1 = unique_labels[i]
                lbl2 = unique_labels[j]
                
                sim = float(np.dot(cluster_profiles[lbl1], cluster_profiles[lbl2]))
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (lbl1, lbl2)
                    
        # If the best pair exceeds the merge threshold, merge them!
        if best_pair and best_sim >= CLUSTER_MERGE_THRESHOLD:
            lbl1, lbl2 = best_pair
            print(f"[DEBUG] Merging established clusters: Cluster {lbl1} and Cluster {lbl2} (Centroid similarity: {best_sim:.4f} >= {CLUSTER_MERGE_THRESHOLD})")
            
            # Reassign all faces of cluster 2 to cluster 1
            for fid in assignments:
                if assignments[fid] == lbl2:
                    assignments[fid] = lbl1
        else:
            break
            
    # Remap final labels to continuous 0, 1, 2, ...
    unique_labels = sorted(list(set(assignments.values())))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for fid in assignments:
        assignments[fid] = label_map[assignments[fid]]
        
    return assignments
