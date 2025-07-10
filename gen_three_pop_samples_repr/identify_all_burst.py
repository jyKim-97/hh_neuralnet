import extract_burstprobs as eb
import identify_burstmasks as ibm


if __name__ == "__main__":
    for cid in range(1, 11):
        print("Cluster ID: %d"%(cid))
        
        eb.main(
            fout=f"./postdata/mfop/burst_props_{cid}.pkl",
            cid=cid,
        )
        
        ibm.main(
            cid=cid,
            tmin_discon=0.1,
            fout=f"./postdata/mfop/motif_info_{cid}.pkl",
            export_spectrum=True,
            export_spectrum_dir="./postdata/mfop/spec_summary"
        )