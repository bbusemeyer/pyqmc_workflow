import qmc_manager
import hartree_fock
import numpy as np
from cluster_setup import setup_cluster

cluster_kws=dict(run_strategy = "local",ncore=2)
jastrow_kws=dict(ion_cusp=True)

rule GEOMETRY:
    output: "r{bond}/geometry.xyz"
    run:
        with open(output[0],'w') as f:
            f.write(f"H 0. 0. 0.; H 0. 0. {wildcards.bond}")


rule HARTREE_FOCK:
    input: "r{bond}/geometry.xyz"
    output: hf="r{bond}/hf.chk"
    run:
        with open(input[0],'r') as f:
            txt = f.read()
        client, cluster = setup_cluster(**cluster_kws)
        x=client.submit(hartree_fock.hf,txt,output.hf)
        x.result()
        client.close()

rule OPTIMIZE: 
    input: hf="r{bond}/hf.chk"
    output: 'r{bond}/eigenstate0.chk'
    run: 
        client, cluster = setup_cluster(**cluster_kws)
        mc_calc = qmc_manager.QMCManager(input.hf, client, cluster_kws['ncore'], jastrow_kws=jastrow_kws)
        mc_calc.optimize(hdf_file=output[0], verbose=True)
        client.close()

rule DMC:
    input: hf = "r{bond}/hf.chk", opt = "r{bond}/eigenstate0.chk"
    output: "r{bond}/dmc{tstep}.chk"
    run:
        client, cluster = setup_cluster(**cluster_kws)
        mc_calc = qmc_manager.QMCManager(input.hf, client, cluster_kws['ncore'], jastrow_kws=jastrow_kws)
        mc_calc.read_wf(input.opt)
        mc_calc.dmc(hdf_file=output[0], tstep=float(wildcards.tstep), verbose=True)
        client.close()

