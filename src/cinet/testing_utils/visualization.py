from cinet import *
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import plotnine as p9
import patchworklib as pw
import seaborn as sns

def obtain_distr(drug_name):
    train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    train_file = drug_name + ".csv"
    whole_train_path = train_path + train_file
    table = pd.read_csv(whole_train_path).set_index('cell_line')
    train_y = table.iloc[:,0]  # y contains the response data (AAC)

    gcsi_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    gcsi_file = drug_name + ".csv"
    whole_gcsi_path = gcsi_path + gcsi_file
    table = pd.read_csv(whole_gcsi_path).set_index('cell_line')
    gcsi_y = table.iloc[:,0]  # y contains the response data (AAC)

    gdsc_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    gdsc_file = drug_name + ".csv"
    whole_gdsc_path = gdsc_path + gdsc_file
    table = pd.read_csv(whole_gdsc_path).set_index('cell_line')
    gdsc_y = table.iloc[:,0]  # y contains the response data (AAC)

    return (train_y, gcsi_y, gdsc_y)

def plot_distribution(drug_name):
    # train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    train_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_data/"
    train_file = drug_name + ".csv"
    whole_train_path = train_path + train_file
    table = pd.read_csv(whole_train_path).set_index('cell_line')
    train_y = table.iloc[:,0]  # y contains the response data (AAC)
    plt.subplot(3,1,1)
    plt.hist(train_y, bins=300, range=(0,1), color="lightblue")
    plt.legend(['CCLE'])
    plt.grid(True)
    plt.xlim(left=0, right=1)

    # gcsi_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    gcsi_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/gCSI_Test_Data/"
    gcsi_file = drug_name + ".csv"
    whole_gcsi_path = gcsi_path + gcsi_file
    table = pd.read_csv(whole_gcsi_path).set_index('cell_line')
    gcsi_y = table.iloc[:,0]  # y contains the response data (AAC)
    plt.subplot(3,1,2)
    plt.hist(gcsi_y, bins=300, range=(0,1), color="orange")
    plt.legend(['gCSI'])
    plt.grid(True)
    plt.ylabel("Counts")
    plt.xlim(left=0, right=1)

    # gdsc_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    gdsc_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/GDSC_Test_Data/"
    gdsc_file = drug_name + ".csv"
    whole_gdsc_path = gdsc_path + gdsc_file
    table = pd.read_csv(whole_gdsc_path).set_index('cell_line')
    gdsc_y = table.iloc[:,0]  # y contains the response data (AAC)
    plt.subplot(3,1,3)
    plt.hist(gdsc_y, bins=300, range=(0,1), color="red")
    plt.legend(['GDSC'])
    plt.grid(True)
    plt.xlabel("Area Above the Curve (AAC)")
    plt.xlim(left=0, right=1)
    
    plt.suptitle("Filtered distribution of responses to " + drug_name)
    plt.show()

def plotnine_distribution(drug_name):
    train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    test_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/"
    train_file = drug_name + ".csv"
    gcsi_file = "gCSI_Test_Data/" + drug_name + ".csv"
    gdsc_file = "GDSC_Test_Data/" + drug_name + ".csv"
    train_table = pd.read_csv(train_path + train_file)
    gcsi_table = pd.read_csv(test_path + gcsi_file)
    gdsc_table = pd.read_csv(test_path + gdsc_file)
    tables = [(train_table, "CCLE"), (gcsi_table, "gCSI"), (gdsc_table, "GDSC")]
    result = pd.concat([df.assign(dataset=k) for (df, k) in tables])
    fig = (p9.ggplot(result, p9.aes(x='target', color='dataset', fill='dataset')) + p9.geom_density(alpha=0.1)
          + p9.scales.xlim(0,1)
          + p9.labels.ggtitle(drug_name)
          + p9.labels.xlab("Area Above the Curve (AAC)"))
    return fig

def seaborn_distribution(drug_name):
    train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    test_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/"
    train_file = drug_name + ".csv"
    gcsi_file = "gCSI_Test_Data/" + drug_name + ".csv"
    gdsc_file = "GDSC_Test_Data/" + drug_name + ".csv"
    train_table = pd.read_csv(train_path + train_file)
    gcsi_table = pd.read_csv(test_path + gcsi_file)
    gdsc_table = pd.read_csv(test_path + gdsc_file)
    tables = [(train_table, "CCLE"), (gcsi_table, "gCSI"), (gdsc_table, "GDSC")]
    result = pd.concat([df.assign(dataset=k) for (df, k) in tables])
    fig = sns.displot(result, x="target", hue="dataset", kind="kde", fill=True, legend=False).set(title=drug_name, xlim=(0,1), xlabel="", ylabel="")
    # fig = sns.displot(train_table, x="target", kind="kde", fill=True, legend=True).set(title=drug_name, xlim=(-0.1,1.2), xlabel="AAC", ylabel="Density")
    return fig

def show_all_distributions():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_dat/"
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        drug_name = ""
        idx = 17
        while filename[idx] != "_":
            drug_name = drug_name + filename[idx]
            idx = idx + 1
        if os.path.isfile(f):
            fig = plotnine_distribution(drug_name)
            print(fig)

def all_distributions_subplotnine():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_data/"
    figures = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        drug_name = ""
        idx = 17
        while filename[idx] != "_":
            drug_name = drug_name + filename[idx]
            idx = idx + 1
        if os.path.isfile(f):
            fig = plotnine_distribution(drug_name)
            figures.append(pw.load_ggplot(fig))
    hor_figs = []
    for idx in range(3):
        hor_fig = figures[idx*7]
        for idx2 in range(1,7):
            fig = figures[idx*7 + idx2]
            hor_fig = hor_fig | fig
        hor_figs.append(hor_fig)
    res_fig = hor_figs[0]
    for idx in range(1,3):
        res_fig = res_fig / hor_figs[idx]
    res_fig.savefig("plotnine.png")

def all_distributions_subseaborn():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_data/"
    figures = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        drug_name = filename[0:-4]
        # idx = 0
        # while (filename[idx] != "-" or idx == 1 or idx == 4 or idx == 2):
        #    drug_name = drug_name + filename[idx]
        #     idx = idx + 1
        print(drug_name)
        if os.path.isfile(f):
            fig = seaborn_distribution(drug_name)
            figures.append(pw.load_seaborngrid(fig))
    hor_figs = []
    for idx in range(7):
        hor_fig = figures[idx*3]
        for idx2 in range(1,3):
            if len(figures) > (idx*3+idx2):
                fig = figures[idx*3 + idx2]
                hor_fig = hor_fig | fig
        hor_figs.append(hor_fig)
    res_fig = hor_figs[0]
    for idx in range(1,7):
        res_fig = res_fig / hor_figs[idx]
    res_fig.savefig("seaborn-vertical.png")

def check_datasets(drugs):
    # train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # train_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_data/"
    # train_file = drug_name + ".csv"
    # whole_train_path = train_path + train_file
    # train_table = pd.read_csv(whole_train_path)
    # train_cells = train_table.iloc[:,0]

    gcsi_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    gdsc_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    xlim = -0.05
    index = 1
    for drug_name in drugs:

        # gcsi_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/gCSI_Test_Data/"
        gcsi_file = drug_name + ".csv"
        whole_gcsi_path = gcsi_path + gcsi_file
        gcsi_table = pd.read_csv(whole_gcsi_path)
        gcsi_cells = gcsi_table.iloc[:,0]

        # gdsc_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/GDSC_Test_Data/"
        gdsc_file = drug_name + ".csv"
        whole_gdsc_path = gdsc_path + gdsc_file
        gdsc_table = pd.read_csv(whole_gdsc_path)
        gdsc_cells = gdsc_table.iloc[:,0]

        test_cells = pd.merge(gdsc_cells, gcsi_cells)
        # gcsi_train_cells = pd.merge(gcsi_cells, train_cells)
        # gdsc_train_cells = pd.merge(gdsc_cells, train_cells)
        # shared_cells = pd.merge(test_cells, train_cells)

        # shared_train_response = pd.merge(shared_cells, train_table)['target']
        # shared_gcsi_response = pd.merge(shared_cells, gcsi_table)['target']
        # shared_gdsc_response = pd.merge(shared_cells, gdsc_table)['target']

        test_gcsi_response = pd.merge(test_cells, gcsi_table)['target']
        test_gdsc_response = pd.merge(test_cells, gdsc_table)['target']

        # gcsi_train_gcsi_response = pd.merge(gcsi_train_cells, gcsi_table)['target']
        # gcsi_train_train_response = pd.merge(gcsi_train_cells, train_table)['target']

        # gdsc_train_gdsc_response = pd.merge(gdsc_train_cells, gdsc_table)['target']
        # gdsc_train_train_response = pd.merge(gdsc_train_cells, train_table)['target']

        # r2_gcsi_train_local = r2_score(gcsi_train_train_response, gcsi_train_gcsi_response)
        # corr_gcsi_train_local = gcsi_train_train_response.corr(gcsi_train_gcsi_response)
        # corr_gcsi_train_global = shared_train_response.corr(shared_gcsi_response)
        # corr_gdsc_train_local = gdsc_train_train_response.corr(gdsc_train_gdsc_response)
        # corr_gdsc_train_global = shared_train_response.corr(shared_gdsc_response)
        corr_test_local = test_gcsi_response.corr(test_gdsc_response)
        # corr_test_global = shared_gcsi_response.corr(shared_gdsc_response)
        """
        plt.subplot(1,3,1)
        plt.plot(gcsi_train_train_response, gcsi_train_gcsi_response, 'bx', label='Local')
        plt.plot(shared_train_response, shared_gcsi_response, 'ro', label='Global')
        plt.plot([xlim,1], [xlim,1], '--k')
        plt.text(0.3, 0.92, 'Pearson local = ' + str(corr_gcsi_train_local.round(3)) + "\nPearson global = " + str(corr_gcsi_train_global.round(3)) , bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 0.1, 'boxstyle': 'round'})
        plt.grid(True)
        plt.xlabel('CCLE')
        plt.ylabel('gCSI')
        plt.legend()
        plt.xlim(xlim,1)
        plt.ylim(xlim,1)

        plt.subplot(1,3,2)
        plt.plot(gdsc_train_train_response, gdsc_train_gdsc_response, 'bx', label='Local')
        plt.plot(shared_train_response, shared_gdsc_response, 'ro', label='Global')
        plt.plot([xlim,1], [xlim,1], '--k')
        plt.text(0.3, 0.92, 'Pearson local = ' + str(corr_gdsc_train_local.round(3)) + "\nPearson global = " + str(corr_gdsc_train_global.round(3)) , bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 0.1, 'boxstyle': 'round'})
        plt.grid(True)
        plt.xlabel('CCLE')
        plt.ylabel('GDSC')
        plt.legend()
        plt.xlim(xlim,1)
        plt.ylim(xlim,1)
        """
        plt.subplot(1,3,index)
        if index == 1:
            plt.plot(test_gcsi_response, test_gdsc_response, 'bx')
            plt.ylabel('GDSC (AAC)')
        elif index == 3:
            plt.plot(test_gcsi_response, test_gdsc_response, 'bx', label='Cell-lines')
            plt.legend(loc='lower right')
        else:
            plt.plot(test_gcsi_response, test_gdsc_response, 'bx')
        # plt.plot(shared_gcsi_response, shared_gdsc_response, 'ro', label='Global')
        plt.text(0.0, 0.94, 'Pearson = ' + str(corr_test_local.round(3)) , bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 0.1, 'boxstyle': 'round'})
        plt.plot([xlim,1], [xlim,1], '--k')
        plt.grid(True)
        plt.xlabel('gCSI (AAC)')
        plt.title(drug_name)
        plt.xlim(xlim,1)
        plt.ylim(xlim,1)
        index += 1
    plt.show()

def method_validation_plots():
    path = "results/Method-Validation/"
    drugs = ["AZD7762", "Lapatinib", "Vorinostat"]
    idx = 0
    fake_data_good = []
    fake_data_bad = []
    deltas = np.append(np.linspace(0,0.1,11), [0.15, 0.2])
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            results = np.load(f)
            drug = drugs[idx]
            res_drug_good = []
            res_drug_bad = []
            for elem in results[idx]:
                res_drug_good.append(elem[0])
                res_drug_bad.append(elem[1])
            idx += 1
            plt.figure(figsize=(5,4))
            plt.plot(deltas, res_drug_good,'o-')
            plt.plot(deltas, res_drug_bad,'o-')
            plt.legend(["Valid Pairs", "Random Pairs"], loc='lower center')
            plt.xlabel("Delta")
            plt.ylabel("Concordance Index")
            plt.title(drug)
            plt.grid(True)
            plt.ylim([0.3, 0.7])
            plt.show()
            fake_data_good.append(res_drug_good)
            fake_data_bad.append(res_drug_bad)
    fake_data_good = np.array(fake_data_good)
    fake_data_bad = np.array(fake_data_bad)
    cdf = pd.DataFrame()
    for delta_idx in range(0,13):
        good_data = [fake_data_good[0][delta_idx], fake_data_good[1][delta_idx], fake_data_good[2][delta_idx]]
        bad_data = [fake_data_bad[0][delta_idx], fake_data_bad[1][delta_idx], fake_data_bad[2][delta_idx]]
        df = pd.DataFrame(data={'Valid': good_data, 'Random': bad_data}).assign(Delta=deltas[delta_idx])
        cdf = pd.concat([cdf, df])    
    mdf = pd.melt(cdf, id_vars=['Delta'], var_name=['Pairs'])
    plt.figure(figsize=(8,5))
    ax = sns.boxplot(x="Delta", y="value", hue="Pairs", data=mdf)   
    plt.grid(True)
    plt.title("Average across drugs")
    plt.ylabel("Concordance Index")
    plt.xlabel("Deltas")
    plt.show()

def model_inference_plots():
    path = "results/Model-Inference/"
    general_results = [[(0.5241657077100115, 0.5247081400927555), (0.5724971231300345, 0.5269870462178154), (0.5, 0.5), (0.48590333716915995, 0.5), (0.4982738780207135, 0.5), (0.5388377445339471, 0.5077562769870462), (0.5650172612197929, 0.5302254917639533), (0.5417146144994246, 0.5100751639213178), (0.5687571921749137, 0.530065568527107), (0.4982738780207135, 0.5013193667039821), (0.5468929804372842, 0.5239285143131297), (0.5546605293440736, 0.5256276987046218), (0.523590333716916, 0.5393211258595874)], [(0.5398274987316083, 0.5480007172314865), (0.5824454591577879, 0.5575757575757576), (0.5946220192795535, 0.5699121391429084), (0.5276509386098427, 0.5225031378877533), (0.6341958396752917, 0.5799533799533799), (0.5938609842719432, 0.5673480365788058), (0.5545408422120751, 0.5569840415994262), (0.5540334855403348, 0.5390353236507083), (0.5291730086250634, 0.5314685314685315), (0.5961440892947742, 0.5675094136632598), (0.5078640284119736, 0.48933118163887396), (0.5, 0.5), (0.45712836123795025, 0.5355567509413663)], [(0.6541218637992832, 0.5580720092915215), (0.6610343061955966, 0.5456114152978264), (0.6733230926779313, 0.5623361539737847), (0.6103430619559652, 0.5157126265140203), (0.6625704045058883, 0.5448481831757093), (0.6589861751152074, 0.5341297494607599), (0.5870455709165386, 0.5291189646590343), (0.6589861751152074, 0.541065206570433), (0.6625704045058883, 0.562020905923345), (0.6763952892985151, 0.5535755765720922), (0.6541218637992832, 0.5383109341297495), (0.5704045058883769, 0.5169238427078149), (0.6141833077316948, 0.534478181516509)]]
    drugs = ["Crizotinib", "Dabrafenib", "Docetaxel", "Lapatinib", "Vorinostat"]
    idx = 0
    common_data = []
    exclusive_data = []
    deltas = np.append(np.linspace(0,0.1,11), [0.15, 0.2])
    first = True
    for results in general_results:
        common_drug = []
        exclusive_drug = []
        drug = drugs[idx]
        for elem in results:
            common_drug.append(elem[0])
            exclusive_drug.append(elem[1])
        plt.figure(figsize=(5,4))
        plt.plot(deltas, common_drug,'o-')
        plt.plot(deltas, exclusive_drug,'o-')
        plt.legend(["Common cell-lines", "Exclusive GDSC-v2 cell-lines"], loc='lower center')
        plt.xlabel("Delta")
        plt.ylabel("Concordance Index")
        plt.title(drug)
        plt.grid(True)
        plt.ylim([0.3, 0.7])
        plt.show()
        common_data.append(common_drug)
        exclusive_data.append(exclusive_drug)
        idx += 1
    common_data = np.array(common_data)
    exclusive_data = np.array(exclusive_data)
    cdf = pd.DataFrame()
    for delta_idx in range(0,13):
        good_data = [common_data[0][delta_idx], common_data[1][delta_idx], common_data[2][delta_idx]]
        bad_data = [exclusive_data[0][delta_idx], exclusive_data[1][delta_idx], exclusive_data[2][delta_idx]]
        df = pd.DataFrame(data={'Common': good_data, 'Exclusive': bad_data}).assign(Delta=deltas[delta_idx])
        cdf = pd.concat([cdf, df])    
    mdf = pd.melt(cdf, id_vars=['Delta'], var_name=['Cell-lines'])
    plt.figure(figsize=(8,5))
    ax = sns.boxplot(x="Delta", y="value", hue="Cell-lines", data=mdf)   
    plt.grid(True)
    plt.title("Average across drugs")
    plt.ylabel("Concordance Index")
    plt.xlabel("Deltas")
    plt.show()

def correlation_plots():
    drugs = ["AZD7762", "AZD8055", "Dabrafenib", "Erlotinib", "Gemcitabine", "Lapatinib"]
    drugs_one = ["Dabrafenib", "Erlotinib", "Lapatinib"]
    drugs_two = ["AZD7762", "AZD8055", "Gemcitabine"]
    path = "results/Correlation/"
    gcsi = []
    gdsc = []
    deltas = np.append(np.linspace(0,0.1,11), [0.15, 0.2])
    idx = 0
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        gcsi_drug = []
        gdsc_drug = []
        # checking if it is a file
        if os.path.isfile(f):
            results = np.load(f)
            for elem in results:
                gcsi_drug.append(elem[0])
                gdsc_drug.append(elem[1])
            gcsi.append(gcsi_drug)
            gdsc.append(gdsc_drug)
            idx += 1
    gcsi = np.array(gcsi)
    gdsc = np.array(gdsc)
    cdf = pd.DataFrame()
    for delta_idx in range(0,13):
        one = [gdsc[0][delta_idx], gdsc[1][delta_idx], gdsc[4][delta_idx]]
        two = [gdsc[2][delta_idx], gdsc[3][delta_idx], gdsc[5][delta_idx]]
        df = pd.DataFrame(data={'1': one, '2': two}).assign(Delta=deltas[delta_idx])
        cdf = pd.concat([cdf, df])    
    mdf = pd.melt(cdf, id_vars=['Delta'], var_name=['Clusters'])
    plt.figure(figsize=(8,5))
    ax = sns.boxplot(x="Delta", y="value", hue="Clusters", data=mdf)   
    plt.grid(True)
    plt.title("GDSC-v2")
    plt.ylabel("Concordance Index")
    plt.xlabel("Deltas")
    plt.show()

def arch_plots():
    results_one = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults1.npy")
    results_two = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults2.npy")
    results_three = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults3.npy")
    # results_four = np.load("results/Architecture/ArchitectureExperimentResults4.npy")
    # results_five = np.load("results/Architecture/ArchitectureExperimentResults5.npy")
    # results_six = np.load("results/Architecture/ArchitectureExperimentResults6.npy")
    """
    df = pd.DataFrame(np.concatenate([results_one[0], results_four[0]], axis=None))
    df2 = pd.DataFrame(np.concatenate([results_two[0], results_five[0]], axis=None))
    df3 = pd.DataFrame(np.concatenate([results_two[1], results_five[1]], axis= None))
    df4 = pd.DataFrame(np.concatenate([results_three[0], results_six[0]], axis=None))
    df5 = pd.DataFrame(np.concatenate([results_three[1], results_six[1]],axis=None))
    """
    df = pd.DataFrame(np.average(results_one[0],axis=0))
    df2 = pd.DataFrame(np.average(results_two[0],axis=0))
    df3 = pd.DataFrame(np.average(results_two[1],axis=0))
    df4 = pd.DataFrame(np.average(results_three[0],axis=0))
    df5 = pd.DataFrame(np.average(results_three[1],axis=0))
    res = pd.concat([df,df2,df3,df4,df5], axis=1)
    res.columns = ["2 Layers", "3 Small Layers", "3 Big Layers", "4 Small Layers", "4 Big Layers"]
    # coloring/customization
    my_pal = ['#4daf4a','#ff7f00','#ff7f00','#377eb8','#377eb8']
    plt.rcParams.update({'font.size': 14})
    # plt.rcParams["font.family"] = "Avenir"
    # plotting
    # plt.axvline(x=your_number,linestyle='--',color='black')
    ax = sns.violinplot(data=res,orient='h', palette=my_pal)
    # sns.stripplot(data=test_df,orient='h',edgecolor='black', linewidth=1, palette=['white'] * 4,ax=ax)
    # custom legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label='2 Layers', markeredgecolor='k',markerfacecolor='#4daf4a', markersize=10,),
                    plt.Line2D([0], [0], marker='s', color='w', label='3 Layers', markeredgecolor='k',markerfacecolor='#ff7f00', markersize=10),
                    plt.Line2D([0], [0], marker='s', color='w', label='4 Layers', markeredgecolor='k',markerfacecolor='#377eb8', markersize=10)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1),fontsize=11, loc='upper left')
    plt.xlabel('Concordance Index (C-Index)')
    plt.xlim([0.35,1])
    plt.ylabel('Architecture')
    plt.show()

def arch_heatmap():
    results_one = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults1.npy")
    results_two = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults2.npy")
    results_three = np.load("results/NEW-Architecture/NEW-ArchitectureExperimentResults3.npy")
    # results_four = np.load("results/Architecture/ArchitectureExperimentResults4.npy")
    # results_five = np.load("results/Architecture/ArchitectureExperimentResults5.npy")
    # results_six = np.load("results/Architecture/ArchitectureExperimentResults6.npy")
    deltas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    """
    df = pd.DataFrame(np.average(np.concatenate([results_one[0], results_four[0]]), axis=0))
    df2 = pd.DataFrame(np.average(np.concatenate([results_two[0], results_five[0]]), axis=0))
    df3 = pd.DataFrame(np.average(np.concatenate([results_two[1], results_five[1]]), axis=0))
    df4 = pd.DataFrame(np.average(np.concatenate([results_three[0], results_six[0]]), axis=0))
    df5 = pd.DataFrame(np.average(np.concatenate([results_three[1], results_six[1]]),axis=0))
    """
    df = pd.DataFrame(np.average(results_one[0],axis=0))
    df2 = pd.DataFrame(np.average(results_two[0],axis=0))
    df3 = pd.DataFrame(np.average(results_two[1],axis=0))
    df4 = pd.DataFrame(np.average(results_three[0],axis=0))
    df5 = pd.DataFrame(np.average(results_three[1],axis=0))
    """
    df = pd.DataFrame(results_one[0][2])
    df2 = pd.DataFrame(results_two[0][2])
    df3 = pd.DataFrame(results_two[1][2])
    df4 = pd.DataFrame(results_three[0][2])
    df5 = pd.DataFrame(results_three[1][2])
    """
    res = pd.concat([df,df2,df3,df4,df5], axis=1)
    avg = res.mean(axis=1)
    res.columns = ["2 Layers", "3 Small Layers", "3 Big Layers", "4 Small Layers", "4 Big Layers"]
    res.insert(0, column='Deltas', value=deltas)
    res.set_index('Deltas', inplace=True)
    plt.figure(figsize=(7,5))
    """
    sns.heatmap(res, vmin=0.35, vmax=1.0, annot=True, fmt=".3f", xticklabels=1, yticklabels=1)
    plt.xticks(rotation=0)
    plt.title("Ibrutinib")
    """
    sns.boxplot(data=res.transpose())
    plt.grid(True)
    plt.ylim([0.45,0.7])
    plt.xlabel("Deltas")
    plt.ylabel("Concordance Index")
    plt.title("Average across drugs")
    plt.show()

def less_genes_heatmap():
    less_genes_exp_azd = np.load("LessGenesExperiment-AZD7762.npy")
    less_genes_exp_dab = np.load("LessGenesExperiment-Dabrafenib.npy")
    file = open('LessGenesExperiment.json')
    params = json.load(file)
    azd_50 = less_genes_exp_azd[0:5]
    azd_100 = less_genes_exp_azd[5:10]
    azd_200 = less_genes_exp_azd[10:15]
    dab_50 = less_genes_exp_dab[0:5]
    dab_100 = less_genes_exp_dab[5:10]
    dab_200 = less_genes_exp_dab[10:15]
    res_50 = np.average([azd_50, dab_50], axis=0)
    res_100 = np.average([azd_100, dab_100], axis=0)
    res_200 = np.average([azd_200, dab_200], axis=0)
    df = pd.DataFrame(data={'Deltas': params['deltas'], '50 Genes': res_50, '100 Genes': res_100, '200 Genes': res_200})
    df.set_index('Deltas', inplace=True)
    sns.heatmap(df, vmin=0.35, vmax=1.0, annot=True, fmt=".3f", xticklabels=1, yticklabels=1)
    plt.xticks(rotation=0)
    plt.title("Average across drugs")
    plt.show()

def mrmr_plots():
    drugs = ["AZD7762", "Dabrafenib", "Ibrutinib"]
    archs = ["(32, 32, 0, 0)", "(64, 64, 0, 0)", "(128, 128, 0, 0)"]
    file = open('results/mRMR/mRMRExperiment.json')
    params = json.load(file)
    for drug in drugs:
        data = []
        for arch in archs:
            results = np.load("results/mRMR/mRMRExperiment-" + drug + "-" + arch + ".npy")
            # df = pd.DataFrame(data={'Deltas': params['deltas'], '50 Genes': results[0], 
            #                        '100 Genes': results[1], '200 Genes': results[2]})
            # df.set_index('Deltas', inplace=True)
            # sns.heatmap(df, vmin=0.35, vmax=1.0, annot=True, fmt=".3f", xticklabels=1, yticklabels=1)
            # plt.xticks(rotation=0)
            # plt.title(drug + " " + arch)
            # plt.show()
            data.append([results])
        # avg_index = np.average(data, axis=2)
        # df = pd.DataFrame(data={'Deltas': params['deltas'], '50 Genes': avg_index[0][0], 
        #                        '100 Genes': avg_index[1][0], '200 Genes': avg_index[2][0]})
        # df.set_index('Deltas', inplace=True)
        # coloring/customization
        # my_pal = ['#4daf4a','#ff7f00','#377eb8']
        # plt.rcParams.update({'font.size': 14})
        # plotting
        # ax = sns.violinplot(data=df,orient='h', palette=my_pal)
        # custom legend
        # plt.xlabel('Concordance Index (C-Index)')
        # plt.xlim([0.35,1])
        # plt.ylabel('Number of Genes')
        # plt.title(drug)
        # plt.show()

        avg_index = np.average(data, axis=3)
        df = pd.DataFrame(data={'Number of genes': params['gene indices'], '32': avg_index[0][0], 
                                '64': avg_index[1][0], '128': avg_index[2][0]})
        df.set_index('Number of genes', inplace=True)
        sns.heatmap(df, vmin=0.35, vmax=1.0, annot=True, fmt=".3f", xticklabels=1, yticklabels=1)
        plt.xticks(rotation=0)
        plt.title(drug)
        plt.xlabel("Number of neurons")
        plt.show()

def aac_noise_plots():
    aac_values_one_exp = [0.578, 0.738, 0.572, 0.442, 0.554, 0.562, 0.780, 0.732]
    aac_values_diff_drugs = [0.982, 0.959, 0.902, 0.876, 0.862, 0.849, 0.828, 0.803, 0.738, 0.703, 0.665, 0.654, 0.649, 0.643, 0.629]
    # max1 = float(max(aac_values_one_exp))
    # max2 = float(max(aac_values_diff_drugs))
    # aac_values_one_exp = [x/max1 for x in aac_values_one_exp]
    # aac_values_diff_drugs = [x/max2 for x in aac_values_diff_drugs]
    df = pd.DataFrame(data={'AAC': aac_values_one_exp})
    fwhm1 = 0.365
    fwhm2 = 0.407
    plt.figure(figsize=(5,5))
    fig = sns.displot(df, x="AAC",kind="kde", fill=True, legend=False).set(title="HCC827 treated with Erlotinib - PharmacoDB", xlim=(0,1), xlabel="", ylabel="")
    plt.grid(True)
    plt.xlabel("AAC")
    plt.show()

    df = pd.DataFrame(data={'AAC': aac_values_diff_drugs})
    plt.figure(figsize=(5,5))
    fig = sns.displot(df, x="AAC",kind="kde", fill=True, legend=False).set(title="HCC827 treated with different drugs - PharmacoDB", xlim=(0,1), xlabel="", ylabel="")
    plt.grid(True)
    plt.xlabel("AAC")
    plt.show()

def learn_exp_plots():
    drugs = ["AZD7762", "Lapatinib", "Vorinostat"]
    architectures = ["(128, 256, 128, 0)"]
    file = open('results/mRMR/mRMRExperiment.json')
    params = json.load(file)
    deltas = params["deltas"]
    num_drug = 0
    for drug in drugs:
        elastic_net = np.load("ElasticNet-gCSI-GDSC.npy")
        plt.figure(figsize=(7,5))
        for arch in architectures:
            results = np.load("results/Learn/LearnExperiment-" + drug + "-" + arch + ".npy")
            plt.subplot(1,2,1)
            plt.plot(deltas, results[:,0],'o-')
            plt.subplot(1,2,2)
            plt.plot(deltas, results[:,1],'o-')
        plt.subplot(1,2,1)
        plt.hlines(elastic_net[num_drug][0], deltas[0], deltas[-1], linestyles='dashed', colors="#42a858")
        plt.grid(True)
        plt.legend(["DeepCINET", "ElasticNet"], loc='lower center')
        plt.ylim([0.35, 0.8])
        plt.xlim([min(deltas), max(deltas)])
        plt.title("gCSI")
        plt.xlabel("Delta")
        plt.ylabel("Concordance Index")

        plt.subplot(1,2,2)
        plt.hlines(elastic_net[num_drug][1], deltas[0], deltas[-1], linestyles='dashed', colors="#42a858")
        plt.legend(["DeepCINET", "ElasticNet"], loc='lower center')
        plt.title("GDSC-v2")
        plt.grid(True)
        plt.ylim([0.35, 0.8])
        plt.xlim([min(deltas), max(deltas)])
        plt.xlabel("Delta")

        plt.suptitle(drug)
        plt.show()
        num_drug += 1