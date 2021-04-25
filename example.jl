using Plots

import MovingGaussianMixtures

sample_data = [
0.000177951775538746, 0.00392297222083862, 0.00573375518171434, -0.00215401267990863, 0.000538068347660303,
-0.00143420605437389, 0.000896137706713716, 0.000896941489857404, 0.00107739281386029, 0.0,
-0.00179501035452376, -0.00429492428288084, -0.00658541135833433, 0.00355429554914226, -0.00160099651736935,
-0.000888336204816467, 0.00195538236688178, 0.00445832214171123, 0.00483741873177002, -0.000718132885440779,
0.00377596420953514, 0.00342867790551233, -0.00306831754125248, 0.00397255849263816, 0.0,
-0.000361794504669915, -0.00144587048503415, -0.000361141210136725, -0.00054146738981984, 0.000180456555574247,
-0.00252343320638076, 0.00270392233240091, 0.00488556115164594, 0.00764752452585767, 0.00476365869676501,
0.00147031822114039, 0.00276268708730977, -0.00129020384684025, 0.00535600135308521, 0.000926354860130661,
0.00371402469637161, -0.00278681090717928, -0.00092721378919225, 0.000370782355008043, 0.000556431434184263,
-0.00055643143418421, 0.000556431434184263, 0.00204290162980033, 0.000929973097806059, 0.00447594928538436,
0.00149644622014601, -0.00149644622014607, -0.0102278901832563, 0.0232089773663753, 0.00474519254248678,
-0.000190240655001527, -0.00190041866394127, 0.00152004589404511, 0.000951203343859139, -0.00057083057396285,
-0.000190204470378469, 0.00782820226743366, 0.0042258994890649, 0.00231258534362122, 0.00347893663596757,
-0.000967585948735971, -0.00386100865745943, 0.00173560934514597, 0.00115874868121845, 0.0,
-0.00597246742094086, 0.0113977859017562, -0.00116504867546977, -0.00329361917425058, 0.000967585948736038,
-0.0130796248184689, -0.0146071078637767, 0.00320664237799695, -0.00395965446177131, -0.0018800531952435,
0.00527308189815804, -0.0013208795202859, 0.00605260028192317, 0.000379506645921242, -0.0039776546430593,
-0.00283152619573845, -0.000753721535588146, 0.00150801159586541, -0.00451722958981981, -0.0221013731937308,
0.000183705337156579, -0.000734618949474405, -0.00293309101204326, -0.00073193049928161, -0.00310474178720573,
-0.0032769008023148, 0.0, 0.00711357308880216, -0.00128052702030923, -0.00182648452603419,
0.0, -0.00145878946365998, 0.000729128723516206, -0.00273149582560972, 0.000181867782623716,
0.000727802069971813, -0.00109150456534317, 0.0146522767868704, -0.000184484826646273, 0.000738143602439221,
-0.00772346612385879, -0.000915499468868335, 0.00587373201209392, 0.00943315974911239, -0.000928763882124529,
0.00839009303177396, -0.00392413845613426, 0.00504815506005073, -0.00262074279539635, -0.00298674853354866,
-0.00372093452569019, 0.00278940208757859, 0.00916665290654104, -0.00580906721169516, -0.00149365225678325,
-0.0129751588631334, -0.00587588910566919, 0.00293362879994538, 0.00829727264081383, -0.00535501233509002,
-0.00183992692200718, 0.00997606647844415, 0.031497181728131, 0.00364718707389437, 0.0,
0.00520583456600374, -0.00212416802859568, 0.00483419678746064, -0.00116234030908892, 0.00855537009118863,
0.00195465269427087, -0.0141818792638312, -0.00269697716932484, 0.0190333964890183, 0.00058840837237535,
-0.020006494229477, 0.00675352301577992, -0.0023206353481625, 0.0030953787531742, 0.000581451707437155,
0.00174638639514873, 0.000388500393386961, 0.00917348094033744, 0.0045200037650227, 0.0017742735063647,
-0.00157728739324861, 0.000591191267588397, 0.00652627650127567, -0.000793336019395874, -0.00474684435621956,
0.00554018037561535, -0.0235304974101942, -0.00232288141613964, -0.00135252653214332, -0.00250699196003446,
-0.00652718769663178, -0.00439141517174207, -0.00247360034793817, 0.000950660780790026, -0.0077688690125669,
0.00151114497967001, 0.00682855460685999, 0.00190512536189465, 0.000763067568502649, -0.00266819293039737,
0.00362284694084784, 0.0124941558021624, -0.00674701354659438, -0.000192104505441552, 0.0172434767494189,
0.00136892560734174, 0.00352872352045298, -0.00117762525876351, 0.00157047539149226, -0.00274671548005741,
-0.0109120334509827, 0.00388350002639761, 0.0, -0.000972289818939222, -0.000388651384604448,
-0.00619796677113697, 0.00115919642037643, -0.00328090615641925, 0.00192864090640569, 0.00425614881223867,
0.000193892390331064, -0.00830361051855981, -0.0105213787415328, 0.00324025824339414, -0.00609061646775083,
0.00323102058144654, 0.000571265368292208, -0.00171281800367483, 0.00266565275894659, 0.013629158084423,
-0.00135200406881384, 0.0025123213523506, -0.00173997143946368, 0.00348297565725149, -0.00271002875886502,
0.00077354480747565, 0.0, 0.00893904125683704, 0.00508807359916059, -0.000392310715114109,
-0.0234529598215287, 0.00499328865399085, 0.0025060254079053, 0.0096975158728236, -0.00563600753364361,
0.000775494416530351, -0.00734302816361565, 0.00212007403298724, 0.000192957067651188, -0.00442861992701387,
]

# Don't automatically show plots
default(show=false)

const DISPLAY = false
function do_display(last=false)
	display(plt)

	if ! last
		print("Hit ENTER for next plot: ")
		readline()
	end
end

const N_COMPONENTS = UInt(5)
const WINDOW_SIZE = UInt(10)

@info "Estimating with k-means..."
ret_kmeans = MovingGaussianMixtures.kmeans(sample_data, N_COMPONENTS)
plt = plot(ret_kmeans, sample_data)
savefig(plt, "img/mixture_kmeans.png")

DISPLAY && do_display()

@info "Estimating with EM..."
ret_em = MovingGaussianMixtures.em(sample_data, N_COMPONENTS)
plt = plot(ret_em, sample_data)
savefig(plt, "img/mixture_em.png")

DISPLAY && do_display()

@info "Running MGM..."
ret_mov = MovingGaussianMixtures.em(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
plt = scatter(ret_mov, markersize=2, alpha=.5)
savefig(plt, "img/running_em.png")

DISPLAY && do_display(true)