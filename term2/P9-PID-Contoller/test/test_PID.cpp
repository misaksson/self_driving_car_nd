#include "catch.hpp"
#include "../src/PID.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

TEST_CASE("PID controller should behave as in python lesson example", "[pid]") {

  struct TestElement
  {
    double cte;
    double expected;
  };

  vector<TestElement> testVector;
  testVector.push_back({.cte = 1.0, .expected = 0.20400000000000001});
  testVector.push_back({.cte = 0.9948281010505582, .expected = 0.19142923576598847});
  testVector.push_back({.cte = 0.9796397957432106, .expected = 0.1622609148137744});
  testVector.push_back({.cte = 0.9555156935727638, .expected = 0.13445076656467847});
  testVector.push_back({.cte = 0.923920329590544, .expected = 0.1094135896512779});
  testVector.push_back({.cte = 0.8862006657501809, .expected = 0.08704155997177565});
  testVector.push_back({.cte = 0.8435571145240601, .expected = 0.0671154160273751});
  testVector.push_back({.cte = 0.7970554316171956, .expected = 0.04942890613023965});
  testVector.push_back({.cte = 0.7476399578442852, .expected = 0.033794998608896905});
  testVector.push_back({.cte = 0.6961452028996291, .expected = 0.020042784916327383});
  testVector.push_back({.cte = 0.6433059438543296, .expected = 0.008014644580754425});
  testVector.push_back({.cte = 0.5899662525967978, .expected = -0.0017947252970616084});
  testVector.push_back({.cte = 0.5362263953078759, .expected = -0.01159828926778493});
  testVector.push_back({.cte = 0.4825761449246574, .expected = -0.020129214047619574});
  testVector.push_back({.cte = 0.430007606730328, .expected = -0.025677754692896883});
  testVector.push_back({.cte = 0.3785829037038866, .expected = -0.03101685817970573});
  testVector.push_back({.cte = 0.3285741327050573, .expected = -0.03545651976581493});
  testVector.push_back({.cte = 0.28022580612059755, .expected = -0.039023948615115926});
  testVector.push_back({.cte = 0.2337382637165888, .expected = -0.041804151499698336});
  testVector.push_back({.cte = 0.18927043710522184, .expected = -0.04388148769562539});
  testVector.push_back({.cte = 0.14694404603000066, .expected = -0.04533468311811233});
  testVector.push_back({.cte = 0.10684764452679474, .expected = -0.04623660412460057});
  testVector.push_back({.cte = 0.06904038746347396, .expected = -0.04665446066775539});
  testVector.push_back({.cte = 0.033555516547664865, .expected = -0.0466500543421915});
  testVector.push_back({.cte = 0.000403576153701124, .expected = -0.04628003655083338});
  testVector.push_back({.cte = -0.030424629567221473, .expected = -0.04559617219416336});
  testVector.push_back({.cte = -0.058955326489012805, .expected = -0.044645606487083885});
  testVector.push_back({.cte = -0.08522929512031396, .expected = -0.04347113252235485});
  testVector.push_back({.cte = -0.10929961766976248, .expected = -0.04211145725736568});
  testVector.push_back({.cte = -0.13122963505293228, .expected = -0.0406014637753752});
  testVector.push_back({.cte = -0.1510911012737779, .expected = -0.03897246793766689});
  testVector.push_back({.cte = -0.16896252267827094, .expected = -0.037252467860220885});
  testVector.push_back({.cte = -0.18492766974475217, .expected = -0.03546638493846066});
  testVector.push_back({.cte = -0.1990742493471771, .expected = -0.0336362954641655});
  testVector.push_back({.cte = -0.2114927257753152, .expected = -0.031781652130033845});
  testVector.push_back({.cte = -0.22227527921029377, .expected = -0.029919494954392202});
  testVector.push_back({.cte = -0.2315148908219271, .expected = -0.02806465136997084});
  testVector.push_back({.cte = -0.2393045441533559, .expected = -0.02622992537225636});
  testVector.push_back({.cte = -0.24573653298466525, .expected = -0.02442627577009862});
  testVector.push_back({.cte = -0.25090186640272805, .expected = -0.02266298367958243});
  testVector.push_back({.cte = -0.25488976234476013, .expected = -0.020947809489275736});
  testVector.push_back({.cte = -0.2577872214205854, .expected = -0.019287139591502724});
  testVector.push_back({.cte = -0.26016091060226987, .expected = -0.01923121138782622});
  testVector.push_back({.cte = -0.26157012497096066, .expected = -0.017665910322467226});
  testVector.push_back({.cte = -0.2620176606631814, .expected = -0.01591845207415389});
  testVector.push_back({.cte = -0.2615818089689653, .expected = -0.014227446811876189});
  testVector.push_back({.cte = -0.2603499677338025, .expected = -0.012634509812938483});
  testVector.push_back({.cte = -0.2584067070669446, .expected = -0.0111452262127495});
  testVector.push_back({.cte = -0.25583168891566965, .expected = -0.009758276884906138});
  testVector.push_back({.cte = -0.2526993886553108, .expected = -0.008470768060203741});
  testVector.push_back({.cte = -0.24907916184892534, .expected = -0.007279259708242676});
  testVector.push_back({.cte = -0.24503538962197735, .expected = -0.00618001055965333});
  testVector.push_back({.cte = -0.24062765123269592, .expected = -0.0051690749997274875});
  testVector.push_back({.cte = -0.2359109115985938, .expected = -0.004242366984839357});
  testVector.push_back({.cte = -0.23093571894802553, .expected = -0.0033957122811194285});
  testVector.push_back({.cte = -0.22574840941444585, .expected = -0.002624893363026997});
  testVector.push_back({.cte = -0.2203913159741348, .expected = -0.0019256882186672389});
  testVector.push_back({.cte = -0.21490297949403597, .expected = -0.001293903721260252});
  testVector.push_back({.cte = -0.2093183599597208, .expected = -0.0007254040915871153});
  testVector.push_back({.cte = -0.20366904622383528, .expected = -0.00021613492459422387});
  testVector.push_back({.cte = -0.19798346285951424, .expected = 0.00023785678213844239});
  testVector.push_back({.cte = -0.1922870729237975, .expected = 0.0006404061917737319});
  testVector.push_back({.cte = -0.18660257563485844, .expected = 0.0009952174066890085});
  testVector.push_back({.cte = -0.1809500981454456, .expected = 0.0013058531134111373});
  testVector.push_back({.cte = -0.17534738075482453, .expected = 0.0015757267721407675});
  testVector.push_back({.cte = -0.16980995504409244, .expected = 0.0018080970544438954});
  testVector.push_back({.cte = -0.16435131454635246, .expected = 0.0020060642568301146});
  testVector.push_back({.cte = -0.158983077675139, .expected = 0.002172568440792711});
  testVector.push_back({.cte = -0.1537151427328735, .expected = 0.0023103890714703985});
  testVector.push_back({.cte = -0.14855583490713226, .expected = 0.0024221459474174194});
  testVector.push_back({.cte = -0.14351204523718886, .expected = 0.002510301233063776});
  testVector.push_back({.cte = -0.13858936159766025, .expected = 0.002577162423334483});
  testVector.push_back({.cte = -0.13379219180107724, .expected = 0.0026248860866099864});
  testVector.push_back({.cte = -0.12912387896770688, .expected = 0.0026554822477752524});
  testVector.push_back({.cte = -0.12458680934979872, .expected = 0.002670819287571114});
  testVector.push_back({.cte = -0.12018251282938372, .expected = 0.0026726292478571126});
  testVector.push_back({.cte = -0.11591175633453926, .expected = 0.002662513444776207});
  testVector.push_back({.cte = -0.11177463043931272, .expected = 0.0026419483032105017});
  testVector.push_back({.cte = -0.10777062942788335, .expected = 0.0026122913363933672});
  testVector.push_back({.cte = -0.10389872511460636, .expected = 0.0025747872041331533});
  testVector.push_back({.cte = -0.10015743471884439, .expected = 0.0025305737918651407});
  testVector.push_back({.cte = -0.09654488309742798, .expected = 0.002480688260722024});
  testVector.push_back({.cte = -0.09305885963863629, .expected = 0.0024260730260516564});
  testVector.push_back({.cte = -0.08969687012014783, .expected = 0.0023675816283590546});
  testVector.push_back({.cte = -0.0864561838298419, .expected = 0.002305984466553319});
  testVector.push_back({.cte = -0.08333387624295975, .expected = 0.00224197436868653});
  testVector.push_back({.cte = -0.08032686754225733, .expected = 0.002176171980118805});
  testVector.push_back({.cte = -0.07743195725966165, .expected = 0.00210913095327909});
  testVector.push_back({.cte = -0.07464585530881976, .expected = 0.0020413429269508093});
  testVector.push_back({.cte = -0.0719652096680136, .expected = 0.0019732422863327752});
  testVector.push_back({.cte = -0.06938663096239381, .expected = 0.001905210698048089});
  testVector.push_back({.cte = -0.06690671418352757, .expected = 0.001837581416826539});
  testVector.push_back({.cte = -0.06452205777300539, .expected = 0.001770643362806806});
  testVector.push_back({.cte = -0.06222928028543717, .expected = 0.001704644970316799});
  testVector.push_back({.cte = -0.06002503483470071, .expected = 0.0016397978106300233});
  testVector.push_back({.cte = -0.05790602151588045, .expected = 0.0015762799925819393});
  testVector.push_back({.cte = -0.05586899798403425, .expected = 0.001514239346092884});
  testVector.push_back({.cte = -0.05391078835981929, .expected = 0.0014537963946028637});
  testVector.push_back({.cte = -0.05202829062115417, .expected = 0.0013950471232017482});
  testVector.push_back({.cte = -0.05021848262953996, .expected = 0.0013380655498537062});


  PID pid;
  pid.Init(0.2, 0.004, 3.0);
  for (auto testElement = testVector.begin(); testElement != testVector.end(); ++testElement) {
    const double actual = pid.CalcError(testElement->cte);
    REQUIRE(actual == Approx(testElement->expected));
  }
}
