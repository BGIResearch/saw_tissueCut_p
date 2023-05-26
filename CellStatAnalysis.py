import pandas as pd
import numpy as np
import sys, os
from collections import defaultdict
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt
import seaborn as sns

class cellStatAnalysis():
    def __init__(self, genedf, outpath):
        self.genedf = genedf
        self.labelset = list(set(genedf['label']))
        self.outpath = outpath
        self.outfigpath = os.path.join(outpath, "tissue_fig")
        os.makedirs(self.outfigpath, exist_ok=True)

    def figPlot(self, statdf, typeName):
        figpath = os.path.join(self.outfigpath, "violin_" + typeName + ".png")

        plt.figure()
        sns.violinplot(x=statdf[typeName], orient="v")
        sns.stripplot(x=statdf[typeName], orient="v", jitter=0.4, color="black", size=0.8)
        plt.title(typeName)
        plt.savefig(figpath)

    def CellProcess(self, mergetis):
        ###### Input: merge_GetExp_gene.txt with labels
        ###### Return: statistics analysis result
        unique_labels = self.labelset
        Statdf = pd.DataFrame()
        Statgene = defaultdict(list)

        for i in unique_labels:
            numMT = 0
            numRPL = 0
            df = self.genedf[self.genedf['label'] == i]
            umi = df['MIDCount'].sum()
            gene = len(set(df['geneID']))
            
            reads = mergetis[mergetis['label'] == i]['reads'].sum()
            for g in df['geneID']:
                ## Compute mito gene and rpl gene
                g = g.upper()
                if g.startswith('MT-'):
                    numMT += 1
                if g.startswith('RPL'):
                    numRPL += 1
            dnb = len(set(df['x'].map(str) + "_" + df['y'].map(str)))
            Statgene[i] = [gene, numMT, numRPL, umi, dnb, reads, round(gene/dnb, 2), round(umi/dnb, 2)]

        Statdf = pd.DataFrame.from_dict(Statgene, orient='index', columns=['Genetype', 'MTgene',"RPLgene", 'umi_counts', 'DNB_counts', 'Read_counts', 'GeneperDNB', 'UmiperDNB'])
        Statdf['label'] = unique_labels
        Statdf = Statdf.sort_values(by=['Genetype'], ascending=False)
        filtdf = Statdf[Statdf['umi_counts'] >= 30]
        if len(filtdf) > 0:
            print("number of cells after filtering: ", len(filtdf))
            return filtdf
        else:
            return Statdf

    def StatAnalysis(self, Statdf, labeldf, num_dnb, tot_reads):
        #### Plotting figs
        gene_umi_fig = os.path.join(self.outfigpath, "scatter_cell_umi_gene_counts.png")
        plt.figure(figsize=(5,5))
        # sns.scatterplot(x=df['n_counts'], y=df['n_genes'], edgecolor="gray", color="gray")
        plt.scatter(Statdf['umi_counts'], Statdf['Genetype'], color="gray", edgecolors="gray", s=0.8)
        plt.grid()
        plt.xlabel("n_counts")
        plt.ylabel("n_genes")
        plt.savefig(gene_umi_fig)

        self.figPlot(Statdf, 'Genetype')
        # self.figPlot(Statdf, 'MTgene')
        # self.figPlot(Statdf, 'RPLgene')
        self.figPlot(Statdf, 'umi_counts')
        self.figPlot(Statdf, 'DNB_counts')
        self.figPlot(labeldf, 'CellArea')
        
        #### Get statistic results
        logpath = os.path.join(self.outpath, "TissueCut.log")
        tot_gene_type = len(set(self.genedf['geneID']))
        avggene, medgene = Statdf['Genetype'].mean(), np.median(Statdf['Genetype'])
        print("The average gene: {:.2f} median gene: {}.".format(avggene, medgene))
        avgumi, medumi = Statdf['umi_counts'].mean(), np.median(Statdf['umi_counts'])
        print("The average Umi: {:.2f} median umi: {}.".format(avgumi, medumi))
        avgdnb, meddnb = Statdf['DNB_counts'].mean(), np.median(Statdf['DNB_counts'])
        print("The average DNB: {:.2f} median DNB: {}.".format(avgdnb, meddnb))
        avgmt = Statdf['MTgene'].mean()
        avgRPL = Statdf['RPLgene'].mean()
        geneDNB = Statdf['GeneperDNB'].mean()
        umiDNB = Statdf['UmiperDNB'].mean()
        
        cell_count = self.genedf['label'].max() if not self.genedf.empty else 0
        tot_umi = Statdf['umi_counts'].sum()
        merge_label = pd.merge(Statdf, labeldf, on=['label'], how='inner')
        tot_area = merge_label['CellArea'].sum()
        
        reads_under_cell = Statdf['Read_counts'].sum()
        
        with open(logpath, "w") as log:
            log.write("############## Cell Statistic Analysis ############\n")
            log.write("Total_contour_area: {}\n".format(tot_area))
            log.write("Number_of_DNB_under_cell: {}\nRatio: {:.2f}\n".format(num_dnb, (num_dnb/tot_area)*100))
            log.write("Total_Gene_type: {}\n".format(tot_gene_type))
            log.write("Total_umi_under_cell: %d\n" %(tot_umi))
            log.write("Reads_under_cell: {}\nFraction_Reads_in_Spots_Under_cell: {:.2f}\n".format(reads_under_cell, (reads_under_cell/tot_reads)*100))
            log.write("\n")
            
            log.write("Total_cell_count: %d\n" % (cell_count))
            log.write("Mean_reads: {}\n".format(Statdf['Read_counts'].mean()))
            log.write("Median_reads: {}\n".format(np.median(Statdf['Read_counts'])))
            log.write("Mean_Gene_type_per_cell: {:.2f}\nMedian_Gene_type_per_cell: {}\n".format(avggene, medgene))
            log.write("Mean_Umi_per_cell: {:.2f}\nMedian_Umi_per_cell: {}\n".format(avgumi, medumi))
            log.write("Mean_cell_area: {}\n".format(labeldf['CellArea'].mean()))
            log.write("Mean_DNB_per_cell: {:.2f}\nMedian_DNB_per_cell: {}\n".format(avgdnb, meddnb))
            log.write("Mean_MTgene_counts_per_cell: {:.2f}\n".format(avgmt))
            log.write("Mean_RPLgene_counts_per_cell: {:.2f}\n".format(avgRPL))
            log.write("Mean_gene_type_per_DNB: {:.2f}\n".format(geneDNB))
            log.write("Mean_Umi_per_DNB: {:.2f}\n".format(umiDNB))
