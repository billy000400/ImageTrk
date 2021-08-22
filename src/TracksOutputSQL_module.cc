// This script extracts mcdigi and reco-hits' info to a SQL database.
// Each track is represented by a particleID
// ParticleID comply with: runID.subrunID.eventID.trackID
// where trackID is the cet::map_vector::key of SimParticle
// Currently, the script just supports StrawHit
// Further version will support PanelHit
// Author: Billy Haoyang Li
// Email: li000400@umn.edu

// C++ include
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <unistd.h>
// file system include
#include <string>
#include <boost/algorithm/string.hpp>
// Framework includes
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h" // For DEFINE_ART_MODULE
// Root includes
#include "TH1F.h"
#include "TTree.h"
#include "TMath.h"
// Mu2e Data Products
#include "DataProducts/inc/StrawId.hh"
#include "RecoDataProducts/inc/ComboHit.hh"
#include "MCDataProducts/inc/StrawDigiMC.hh"
#include "MCDataProducts/inc/SimParticle.hh"
// cet lib
#include "cetlib/map_vector.h"
// sqlite include
#include "sqlite3.h"

// string...
using std::to_string;

// CLHEP
using CLHEP::Hep3Vector;

namespace mu2e{
	class TracksOutputSQL : public art::EDAnalyzer {

    public:
			struct Config {
				using Name=fhicl::Name;
				using Comment=fhicl::Comment;

				fhicl::Atom<int> verbose{Name("verbose"), Comment("verbosity level (0-10)"), 0};

				fhicl::Atom<art::InputTag> mcdigisTag{ Name("StrawDigiMCCollection"), Comment("MC digi tag")};
				fhicl::Atom<art::InputTag> chTag{ Name("ComboHitCollection"), Comment("Use strawHit instead of comboHit")};
				fhicl::Sequence<std::string> sourceNames{Name("sourceNames"),Comment("list of files")};
			};

			// the following line is needed to enable art --print-description
			typedef art::EDAnalyzer::Table<Config> Parameters;

    	explicit TracksOutputSQL(const Parameters&);
      void analyze(const art::Event& event) override;
			void beginJob() override;
      void endJob() override;

    private:
			/// fcl config data
			Config _conf;
    	int _verbose;

			//// Database functions
			void create_DB(sqlite3* DB, std::string &sourceName);
			void append_ptcl(sqlite3* DB, int &ptclId, int &run, int &subrun, int &event, int &track, int &pdgId);
			void append_digi(sqlite3* DB, int &digiId,int &ptclId, double &x, double &y, double &z, double &t, double &p, int &station, int &plane, int &panel, int &layer, int &straw, int &uniquePanel, int &uniqueFace, int &uniqueStraw);
			void append_hit(sqlite3* DB, int &ptclId, int &digiId, double &x, double &y, double &z, double &t, int &station, int &plane, int &panel, int &layer, int &straw, int &uniquePanel, int &uniqueFace, int &uniqueStraw);

			// data tag
			const ComboHitCollection* _chcol;
      const StrawDigiMCCollection* _mcdigis;

			// working directory info
			std::string cwd_ideal;
			std::string cwd;
			std::string reco_dir;
			std::string mc_dir;

			// event info
			int fileIndex;
			int runNum;
			int subrunNum;
			int eventNum;

			// track info
			int particleID;
			int strawDigiMCID;

			// Database
			sqlite3* DB;
			std::string db_path;
    };

  // Constructor
  TracksOutputSQL::TracksOutputSQL(const Parameters& conf):
		_conf(conf()),_verbose(conf().verbose()) {

			// initialize file index
			fileIndex = 0;
  }

	// begin job
	void TracksOutputSQL::beginJob(){}

  // end job
  void TracksOutputSQL::endJob(){}

  // Analyzer
  void TracksOutputSQL::analyze(const art::Event& event){

		// initialize indices
		particleID = 0;
		strawDigiMCID = 0;

		// get the source name
		std::string sourceName = _conf.sourceNames(fileIndex);
		create_DB(DB, sourceName);

		// Get run, subrun, and event number
		runNum = event.run();
		subrunNum = event.subRun();
		eventNum = event.event();

    // extract StrawDigiMC collections
    auto mcdH = event.getValidHandle<StrawDigiMCCollection>(_conf.mcdigisTag());
    _mcdigis = mcdH.product();

    // extract ComboHit collections
		auto chH = event.getValidHandle<ComboHitCollection>(_conf.chTag());
		_chcol = chH.product();

		// construct trackId --> particleId map
		cet::map_vector<int> trackID_ParticleID_map;

    // loop through hits
    unsigned nstrs = _mcdigis->size();
		for (unsigned istr=0; istr < nstrs; istr++){

      // Get the SimParticle Pointer from the collection
      StrawDigiMC const& mcdigi = _mcdigis->at(istr);
      StrawEnd itdc;
      art::Ptr<StepPointMC> const& spmcp = mcdigi.stepPointMC(itdc);
			cet::map_vector_key trackKey = spmcp->trackId();
			art::Ptr<SimParticle> const& spp = spmcp->simParticle();
			int spID = spp->pdgId(); // Get SimParticle's pdgID

			if (!trackID_ParticleID_map.has(trackKey)){
				// update map
				particleID ++;
				trackID_ParticleID_map[trackKey] = particleID;
				// append the particle into table
				int trackId = trackKey.asInt();
				append_ptcl(DB, particleID, runNum, subrunNum, eventNum, trackId, spID);
			}

			int particleID = trackID_ParticleID_map[trackKey];

			// Get StrawDigiMC's x, y, z, t, p
			strawDigiMCID++;
      Hep3Vector pos_mc = spmcp->position();
      double x = pos_mc.x();
      double y = pos_mc.y();
      double z = pos_mc.z();
			double t = spmcp->time();
			double p = spmcp->momentum().mag();
			StrawId strawIdMC = spmcp->strawId();
			int stationMC = strawIdMC.getStation();
			int planeMC = strawIdMC.getPlane();
			int panelMC = strawIdMC.getPanel();
			int layerMC = strawIdMC.getLayer();
			int strawMC = strawIdMC.getStraw();
			int uniquePanelMC = strawIdMC.uniquePanel();
			int uniqueFaceMC = strawIdMC.uniqueFace();
			int uniqueStrawMC = strawIdMC.uniqueStraw();

			// append the StrawDigiMC into table
			append_digi(DB, strawDigiMCID, particleID, x, y, z, t, p, stationMC, planeMC, panelMC, layerMC, strawMC, uniquePanelMC, uniqueFaceMC, uniqueStrawMC);

      // Get the reconstructed StrawHits information
			// including:
			// 'x, y, z
			// corrected time
			// strawId
      ComboHit const& ch = _chcol->at(istr);
      Hep3Vector pos_reco = ch.posCLHEP();
      double x_reco = pos_reco.x();
      double y_reco = pos_reco.y();
      double z_reco = pos_reco.z();
			double t_reco = ch.correctedTime();
			StrawId strawId = ch.strawId();
			int station = strawId.getStation();
			int plane = strawId.getPlane();
			int panel = strawId.getPanel();
			int layer = strawId.getLayer();
			int straw = strawId.getStraw();
			int uniquePanel = strawId.uniquePanel();
			int uniqueFace = strawId.uniqueFace();
			int uniqueStraw = strawId.uniqueStraw();

			// append the StrawHit into table
			append_hit(DB, particleID, strawDigiMCID, x_reco, y_reco, z_reco, t_reco, station, plane, panel, layer, straw, uniquePanel, uniqueFace, uniqueStraw);

		}// end of loop of hits

		// close the Database
		sqlite3_close(DB);

		// update file index
		fileIndex += 1;

  }// end of analyzer

	void TracksOutputSQL::create_DB(sqlite3* DB, std::string &sourceName)
	{
		const size_t last_slash_idx = sourceName.find_last_of("\\/");
		if (std::string::npos != last_slash_idx)
		{
				sourceName.erase(0, last_slash_idx + 1);
		}

		std::cerr << "\nCreating a database for " << sourceName << "\n";
    // Below is the directory the script should be called
    // This absolute method should be changed if Billy want it to apply for
    // other people
    cwd_ideal = "/nashome/h/haoyang/mu2e/working/Satellite";

    // Get the current calling directory
		cwd = boost::filesystem::current_path().string();

		// Check the running directory
		if (cwd != cwd_ideal){
			std::cout << "Incorrect working directory \nShould work in: ";
			std::cout << cwd_ideal;
			std::cout << "\nNow working in: " << cwd << "\n";
			std::exit(0);
		}

    // Construct Tracking/tracks if not constructed
    std::string trkDir = cwd_ideal+"tracks/";
    boost::filesystem::create_directory(trkDir);

    // Check if an old database exists
    db_path = cwd_ideal+"/Tracking/tracks/"+sourceName+".db";
    bool oldDbExist = boost::filesystem::exists(db_path);

    // If an old DB exists,
    // delete all old files and csvDir before saving to DB
    (oldDbExist)?boost::filesystem::remove(db_path):true;

		// connect the db
		int error = sqlite3_open(db_path.c_str(), &DB);
		if (error){
			std::cerr << "Failed to create database\n";
			exit(0);
		}

		// create tables
		std::string sql_ptcls, sql_digis, sql_hits;
		char* Err;

		sql_ptcls = "CREATE TABLE Particle(\
			id INTEGER PRIMARY KEY NOT NULL,\
			run INTEGER NOT NULL,\
			subRun INTEGER NOT NULL,\
			event INTEGER NOT NULL,\
			track INTEGER NOT NULL,\
			pdgId INTEGER NOT NULL)";

		sqlite3_exec(DB, sql_ptcls.c_str(), 0, 0, &Err);

		sql_digis = "create table StrawDigiMC(\
			id INTEGER PRIMARY KEY NOT NULL,\
			particle INTEGER NOT NULL,\
			x REAL NOT NULL,\
			y REAL NOT NULL,\
			z REAL NOT NULL,\
			t REAL NOT NULL,\
			p REAL NOT NULL,\
			station INTEGER NOT NULL,\
			plane INTEGER NOT NULL,\
			panel INTEGER NOT NULL,\
			layer INTEGER NOT NULL,\
			straw INTEGER NOT NULL,\
			uniquePanel INTEGER NOT NULL,\
			uniqueFace INTEGER NOT NULL,\
			uniqueStraw INTEGER NOT NULL,\
			foreign key(particle) references Particle(id))";

		sqlite3_exec(DB, sql_digis.c_str(), 0, 0, &Err);

		sql_hits = "create table StrawHit(\
			id INTEGER PRIMARY KEY NOT NULL,\
			particle INTEGER NOT NULL,\
			strawDigiMC INTEGER NOT NULL,\
			x_reco REAL NOT NULL,\
			y_reco REAL NOT NULL,\
			z_reco REAL NOT NULL,\
			t_reco REAL NOT NULL,\
			station INTEGER NOT NULL,\
			plane INTEGER NOT NULL,\
			panel INTEGER NOT NULL,\
			layer INTEGER NOT NULL,\
			straw INTEGER NOT NULL,\
			uniquePanel INTEGER NOT NULL,\
			uniqueFace INTEGER NOT NULL,\
			uniqueStraw INTEGER NOT NULL,\
			foreign key(particle) references Particle(id),\
			foreign key(StrawDigiMC) references StrawDigiMC(id))";

		sqlite3_exec(DB, sql_hits.c_str(), 0, 0, &Err);
	}

	// append particle
	void TracksOutputSQL::append_ptcl(sqlite3* DB, int &ptclId, int &run, int &subrun, int &event, int &track, int &pdgId)
	{
	  std::string sql;
	  sqlite3_stmt* stmt;

	  sql = "append INTO Particle (\
	    id, run, subrun, event, track, pdgId)\
	    VALUES (\
	      ?, ?, ?, ?, ?, ?)";
	  sqlite3_prepare_v2(DB, sql.c_str(), 1000, &stmt, NULL);
	  sqlite3_bind_int(stmt, 1, ptclId);
	  sqlite3_bind_int(stmt, 2, run);
	  sqlite3_bind_int(stmt, 3, subrun);
	  sqlite3_bind_int(stmt, 4, event);
	  sqlite3_bind_int(stmt, 5, track);
		sqlite3_bind_int(stmt, 6, pdgId);
	  sqlite3_step(stmt);
	  sqlite3_finalize(stmt);
	}

	// append StrawDigiMC
	void TracksOutputSQL::append_digi(sqlite3* DB, int &digiId,int &ptclId, double &x, double &y, double &z, double &t, double &p, int &station, int &plane, int &panel, int &layer, int &straw, int &uniquePanel, int &uniqueFace, int &uniqueStraw)
	{
	  std::string sql;
	  sqlite3_stmt* stmt;

	  sql = "append INTO StrawDigiMC(\
	    id, particle, x, y, z, t, p,\
			station, plane, panel, layer, straw,\
			uniquePanel, uniqueFace, uniqueStraw)\
	    VALUES (\
	     ?, ?, ?, ?, ?, ?, ?,\
			 ?, ?, ?, ?, ?,\
			 ?, ?, ?)";
	  sqlite3_prepare_v2(DB, sql.c_str(), -1, &stmt, NULL);
		sqlite3_bind_int(stmt, 1, digiId);
		sqlite3_bind_int(stmt, 2, ptclId);
		sqlite3_bind_double(stmt, 3, x);
		sqlite3_bind_double(stmt, 4, y);
		sqlite3_bind_double(stmt, 5, z);
		sqlite3_bind_double(stmt, 6, t);
		sqlite3_bind_double(stmt, 7, p);
		sqlite3_bind_int(stmt, 8, station);
		sqlite3_bind_int(stmt, 9, plane);
		sqlite3_bind_int(stmt, 10, panel);
		sqlite3_bind_int(stmt, 11, layer);
		sqlite3_bind_int(stmt, 12, straw);
		sqlite3_bind_int(stmt, 13, uniquePanel);
		sqlite3_bind_int(stmt, 14, uniqueFace);
		sqlite3_bind_int(stmt, 15, uniqueStraw);
		sqlite3_step(stmt);
		sqlite3_finalize(stmt);
	}

	// append StrawHit
	void TracksOutputSQL::append_hit(sqlite3* DB, int&ptclId, int &digiId, double &x, double &y, double &z, double &t, int &station, int &plane, int &panel, int &layer, int &straw, int &uniquePanel, int &uniqueFace, int &uniqueStraw)
	{
		std::string sql;
		sqlite3_stmt* stmt;

		sql = "append INTO StrawHit(\
			particle, strawDigiMC, x_reco, y_reco, z_reco, t_reco,\
			station, plane, panel, layer, straw,\
			uniquePanel, uniqueFace, uniqueStraw)\
			VALUES (\
			?, ?, ?, ?, ?, ?,\
			?, ?, ?, ?, ?,\
			?, ?, ?)";
		sqlite3_prepare_v2(DB, sql.c_str(), -1, &stmt, NULL);
		sqlite3_bind_int(stmt, 1, ptclId);
		sqlite3_bind_int(stmt, 2, digiId);
		sqlite3_bind_double(stmt, 3, x);
		sqlite3_bind_double(stmt, 4, y);
		sqlite3_bind_double(stmt, 5, z);
		sqlite3_bind_double(stmt, 6, t);
		sqlite3_bind_int(stmt, 7, station);
		sqlite3_bind_int(stmt, 8, plane);
		sqlite3_bind_int(stmt, 9, panel);
		sqlite3_bind_int(stmt, 10, layer);
		sqlite3_bind_int(stmt, 11, straw);
		sqlite3_bind_int(stmt, 12, uniquePanel);
		sqlite3_bind_int(stmt, 13, uniqueFace);
		sqlite3_bind_int(stmt, 14, uniqueStraw);
		sqlite3_step(stmt);
		sqlite3_finalize(stmt);
	}
} // end namespace mu2e

using mu2e::TracksOutputSQL;
DEFINE_ART_MODULE(mu2e::TracksOutputSQL);
