import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1,101,1)
a = [[[-18.438437591228098], [-50.84384408773324], [0.8606992363929749]], [[-30.15642206175591], [-85.98021213244363], [0.2135435938835144]], [[-21.607252763559853], [-52.75695866185198], [-0.08203568309545517]], [[-62.18035593971704], [-209.97027371014283], [-0.16163454949855804]], [[-32.65894308808329], [-97.38802630950417], [-0.3839912712574005]], [[-14.211606862519265], [-38.32409095104112], [0.023738263174891472]], [[-10.767839242244355], [-23.774477156927407], [-0.06316664814949036]], [[-10.394442728034278], [-31.616893229330515], [0.028973354026675224]], [[-14.241617637252881], [-35.431269070489584], [0.14145474135875702]], [[-8.974414738384743], [-26.63859225656202], [-0.09020376205444336]], [[-14.190219587942357], [-41.045022011992096], [-0.050207894295454025]], [[2.9135064559557784], [0.4457425928122298], [-0.21962414681911469]], [[-4.827967930335484], [-14.828074852107738], [-0.08693695813417435]], [[4.303268487499086], [5.670178071222191], [-0.48680317401885986]], [[0.4069819012514387], [6.391905141985484], [-0.3359951376914978]], [[-6.908227135911912], [-18.301923518210696], [-0.035499606281518936]], [[6.661326862819436], [11.463502635672858], [-0.20789285004138947]], [[-8.715355593588729], [-22.4686614838329], [0.09863504767417908]], [[1.9746888317199616], [-0.7959566396703033], [-0.03412937372922897]], [[-5.909975722597435], [-18.400702777940868], [0.014174111187458038]], [[-3.309492914641001], [-18.762247128250014], [0.05868189409375191]], [[1.878305643502428], [3.1394876134583405], [-0.28748661279678345]], [[-3.92519248612302], [-6.804478557270312], [-0.04895375669002533]], [[10.0235492680033], [20.582480230530656], [-0.030808739364147186]], [[-2.3327531364666987], [-8.877581810278542], [-0.20870131254196167]], [[-3.8886827338404557], [-15.13817864625214], [-0.13115260004997253]], [[12.152329409299432], [14.248769501930308], [-0.06687621027231216]], [[-6.9998486025083135], [-20.328675185409434], [-0.03953821212053299]], [[-8.016505358329592], [-31.327913433231707], [0.03914214298129082]], [[2.856939270096955], [-4.741582581161126], [0.008169679902493954]], [[1.278148412235207], [-8.116916929298347], [0.08166590332984924]], [[16.975052943432708], [34.94760977247682], [-0.08088350296020508]], [[-9.507893919152298], [-31.509536028167616], [0.05893702059984207]], [[1.1877489968388706], [-12.446537792312654], [-0.05949406325817108]], [[-1.9555450630283122], [-22.74839858628776], [-0.07084018737077713]], [[1.733940975387177], [-5.381171614642064], [-0.11512232571840286]], [[7.560784813955427], [18.9015871507553], [-0.05945006012916565]], [[-12.407062592070039], [-49.61408803279519], [-0.027827158570289612]], [[-13.623453426766542], [-38.08087229870213], [-0.023336872458457947]], [[-20.56395684707035], [-55.74423079950494], [0.04977588728070259]], [[-33.432812636866984], [-83.71718324682226], [0.01567424274981022]], [[-12.110660290648749], [-39.964741397526566], [0.19958683848381042]], [[-14.083079597552162], [-51.47688248780029], [0.12320306152105331]], [[4.705522643907004], [11.49617030521057], [0.21029400825500488]], [[-26.755070313050872], [-66.00581201929941], [-0.07511190325021744]], [[-29.731111692540697], [-80.11331270368893], [0.2146262228488922]], [[-35.6387042878976], [-84.59235180338013], [-0.10288875550031662]], [[-25.38451033954218], [-68.80321622487517], [0.1262654811143875]], [[-22.346784558876756], [-57.55242604303328], [0.20301596820354462]], [[-29.234493646167483], [-73.0420182656074], [0.045784350484609604]], [[-27.12265542066691], [-70.02978533649181], [-0.027988525107502937]], [[-25.682915703800113], [-64.30253369501416], [0.05641888454556465]], [[-28.417585183370313], [-70.2663879058421], [-0.1855725646018982]], [[-35.73300287156435], [-121.30680001540479], [0.2557366192340851]], [[-30.888387860466402], [-72.18356848214702], [0.28785738348960876]], [[-38.15964331617382], [-94.25174512370083], [0.33158862590789795]], [[-29.667938410118474], [-75.24679395359037], [0.11645349860191345]], [[-34.07458966479714], [-84.26459604376325], [-0.038501132279634476]], [[-31.43983534369171], [-83.593874243461], [0.125661239027977]], [[-45.72427881057808], [-112.74973797097823], [0.36015528440475464]], [[-39.846714337150004], [-91.94777797845714], [0.3617965877056122]], [[-34.46181218070853], [-90.81792045052151], [0.41820576786994934]], [[-36.36039686592618], [-83.69579401023664], [0.24198421835899353]], [[-25.286364626395404], [-63.45136439482311], [0.026559608057141304]], [[-32.42728542844186], [-86.82978974727152], [-0.02017034962773323]], [[-31.227892555059775], [-82.94021808686378], [-0.03079633228480816]], [[-29.733359460028094], [-69.47383951108847], [0.03285607323050499]], [[-27.44842352346106], [-68.0725009483708], [-0.13474516570568085]], [[-33.219326361446], [-83.52936413061568], [0.039098694920539856]], [[-40.02822756350572], [-110.09719531062085], [0.3473542034626007]], [[-45.06302083926222], [-104.4913769152161], [0.15321430563926697]], [[-31.211450469794254], [-80.90221334921716], [0.14570598304271698]], [[-27.080862088895213], [-64.09871108375492], [0.005719480104744434]], [[-31.105071573710724], [-81.07324721211494], [0.1754666119813919]], [[-45.613843435766], [-126.72056733025761], [0.13702432811260223]], [[-47.57463118078503], [-119.73045739888028], [0.05055882781744003]], [[-50.50562293599356], [-115.60773378177869], [0.18314455449581146]], [[-25.301587016202504], [-63.72523226261113], [0.1648847758769989]], [[-32.39899165177194], [-82.9233227752917], [0.20476210117340088]], [[-37.5194476341821], [-101.47192677808502], [0.2778772711753845]], [[-30.48555921283339], [-73.945649460752], [0.21429097652435303]], [[-41.66226551317855], [-106.11730598062225], [0.12283992022275925]], [[-54.834997951329136], [-147.18665778476083], [0.2656729817390442]], [[-47.62965858803565], [-125.82256182737964], [0.10970232635736465]], [[-30.265528402460756], [-71.7310661945327], [-0.16278468072414398]], [[-37.271878383144106], [-93.83081985149553], [0.30951783061027527]], [[-54.57021799239605], [-125.65040895755695], [-0.08844859153032303]], [[-65.88041303799903], [-161.71435704890723], [0.19221271574497223]], [[-46.64869150098551], [-120.49124046097617], [0.21390260756015778]], [[-39.83693424119586], [-99.66225441466024], [0.24350745975971222]], [[-43.560669338319045], [-104.6258740984388], [0.1643442064523697]], [[-57.46398100439478], [-168.84894127299748], [0.11565117537975311]], [[-48.851306007178856], [-100.85159237677271], [0.14429838955402374]], [[-40.118168356325235], [-101.91310614219528], [0.3091355562210083]], [[-35.84602007326089], [-91.93009639211071], [0.40364912152290344]], [[-34.825935622883435], [-76.85090869622101], [0.2128499299287796]], [[-72.6921799070681], [-241.72135515117756], [0.46002358198165894]], [[-67.94818063820969], [-161.86171737038282], [0.23591962456703186]], [[-57.47634444594918], [-152.15215818731758], [0.528779149055481]], [[-52.95675552206861], [-135.37969502310136], [0.36655136942863464]]]


J=[]
R=[]
e=[]
for i in range(100):
    aJ = a[i][0]
    aR = a[i][1]
    ae = a[i][2]
    J.append(aJ)
    R.append(aR)
    e.append(ae)
# plt.plot(x, J, 'r', label='J Function')
# plt.plot(x, R, 'y', label='R Function')
# plt.plot(x, e, 'b', label='entropy')
fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(x, J, 'r', label='J Function')
ax2.plot(x, R, 'y', label='R Function')
ax3.plot(x, e, 'b', label='entropy')
ax1.set_xlabel('J Function')
ax2.set_xlabel('R Function')
ax3.set_xlabel('entropy')

plt.title('ScratchTiago_Horizon=200_Gamma=0.99_Epoch100_Steps=10000_Steps-test=2000')
plt.legend()
# plt.savefig('feed.pdf')
plt.show()