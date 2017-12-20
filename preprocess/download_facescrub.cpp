#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <exception>
#include <set>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <regex>
#include <cerrno>
#include <cstdlib>
#include <cassert>
#include <string>
#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <curl/curl.h>
#include <dlib/pixel.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/dir_nav/dir_nav_extensions.h>

using namespace std;
using namespace boost::filesystem;

void mkdir_p(const string &folder)
{
    try
    {
        path p = path(folder);
        if (!exists(p)) {
            create_directory(p);
        }
    }
    catch (filesystem_error &e)
    {
        cerr << e.what() << endl;    
    }
}

void urlretrieve(const string &url, const string &local_file, bool do_get=true, bool overwrite=false)
{
    FILE *fpr = NULL;
    if ((!overwrite) && ((fpr = fopen(local_file.c_str(), "rb")) != NULL)) {
        fclose(fpr);
        cerr << "Skipping: already downloaded : " << local_file.c_str() << endl;
        return;    
    } else {
        FILE *fp = fopen(local_file.c_str(), "wb");
        if (fp) {
            CURL *curl = curl_easy_init();
            if (curl) {
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
                if (do_get) {
                    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
                }
                CURLcode res = curl_easy_perform(curl);
                curl_easy_cleanup(curl);
                if (res != CURLE_OK) {
                    cerr << "Failure: curl error for : " << url << " : " << res << endl;
                }
            } else {
                cerr << "Failure: could not initialize curl for : " << url << endl;
            }
            fclose(fp);
        } else {
            cerr << "Failure: could not open : " << local_file << " : " << strerror(errno) << endl;
        }
    }
}


void download(const string &dwldid, const string &images_dwld_dir, const string &bbox_dir, const string &images_list, const string imgfmt="jpg",
             const int class_field=0, const int img_field=1, const int obj_field=2, const int url_field=3, const int bbox_field=4)
{
    const time_t t = time(NULL);
    const clock_t c = clock();
    // const int nproc = omp_get_num_procs();
    const int nproc = 64;
    regex space_re(" ");
    typedef boost::char_separator<char> Separator;
    typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
    Separator sep_tab("\t");
    Separator sep_comma(",");
    vector<string> urls;
    vector<string> imgpaths;
    vector<string> bboxpaths;
    set<string> imgdpaths;
    set<string> bboxdpaths;
    vector<int> x1, y1, x2, y2;

    cout << "Starting " << dwldid << " parse file " << images_list << endl;
    cout.flush();
    ifstream ifs(images_list);
    if (!ifs) {
        cerr << "Could not open image list " << images_list << endl;
        return;
    }
    string line = "";
    getline(ifs, line); // skip header
    while (getline(ifs, line)) {
        Tokenizer tok(line, sep_tab);
        vector<string> vec;
        vec.assign(tok.begin(), tok.end());
        // copy(vec.begin(), vec.end(), ostream_iterator<string>(cout, "|")); cout << endl;
        string classid = regex_replace(vec[class_field], space_re, "_");
        stringstream imgid_ss;
        imgid_ss << setw(9) << setfill('0') << atoll(vec[obj_field].c_str()) << "." << imgfmt;
        stringstream bboxid_ss;
        bboxid_ss << setw(9) << setfill('0') << atoll(vec[obj_field].c_str()) << "." << imgfmt;
        string url = vec[url_field];
        string imgdpath = images_dwld_dir + "/" + classid;
        string bboxdpath = bbox_dir + "/" + classid;
        string imgpath = imgdpath + "/" + imgid_ss.str();
        string bboxpath = bboxdpath + "/" + bboxid_ss.str();
        // cout << url << " ==> " << fpath << endl;
        string bbox_str = vec[bbox_field];
        Tokenizer tok2(bbox_str, sep_comma);
        vector<string> bbox;
        bbox.assign(tok2.begin(), tok2.end());
        // copy(bbox.begin(), bbox.end(), ostream_iterator<string>(cout, "+")); cout << endl;
        x1.push_back(atoi(bbox[0].c_str()));
        y1.push_back(atoi(bbox[1].c_str()));
        x2.push_back(atoi(bbox[2].c_str()));
        y2.push_back(atoi(bbox[3].c_str()));
        urls.push_back(url);
        imgpaths.push_back(imgpath);
        bboxpaths.push_back(bboxpath);
        imgdpaths.insert(imgdpath);
        bboxdpaths.insert(bboxdpath);
    }
    ifs.close();

    mkdir_p(images_dwld_dir);
    mkdir_p(bbox_dir);
    for (const string &dpath: imgdpaths) {
        mkdir_p(dpath);
    }
    for (const string &dpath: bboxdpaths) {
        mkdir_p(dpath);
    }
    cout << "Starting " << dwldid << " queries on " << nproc << " cpu cores" << endl;
    cout.flush();
    const int num_files = int(imgpaths.size());
    #pragma omp parallel for num_threads(nproc)
    for (int i = 0 ; i < num_files; i++) {
        urlretrieve(urls[i], imgpaths[i], false);
        dlib::array2d<dlib::rgb_pixel> img;
        dlib::array2d<dlib::rgb_pixel> crop;
        // cerr << urls[i] << endl;
        // cerr << imgpaths[i] << endl;
        try {
            dlib::load_image(img, imgpaths[i]);
            // cerr << x1[i] << " " << y1[i] << " " << x2[i] << " " << y2[i] << endl;
            if (dlib::file_exists(bboxpaths[i])) {
                cerr << "Skip: crop already present : " << bboxpaths[i] << endl;
            } else {
                dlib::rectangle bbox_(x1[i], y1[i], x2[i], y2[i]);
                dlib::chip_details bbox(bbox_);
                dlib::extract_image_chip(img, bbox, crop);
                dlib::save_jpeg(crop, bboxpaths[i], 100);
            }
        } catch (exception &e) {
          cerr << urls[i] << " :: " << e.what() << endl;
        }
    }

    cout << dwldid << " : " << "time = " << (time(NULL) - t) << " clock = " << float(clock() - c) / CLOCKS_PER_SEC << endl; 
}

void download_facescrub(const string &dset_dir, const string &dwld_dir, const string &subset)
{
    const string base_dwld_dir = dwld_dir + "/" + subset;
    const string images_dwld_dir = base_dwld_dir + "/" + "images";
    const string faces_dir = base_dwld_dir + "/" + "faces";
    const string images_list = dset_dir + "/" + "facescrub" + "_"  + subset + ".txt";
    mkdir_p(base_dwld_dir);
    mkdir_p(images_dwld_dir);
    mkdir_p(faces_dir);
    download(subset, images_dwld_dir, faces_dir, images_list);
}

int main(int argc, char *argv[])
{
    ios::sync_with_stdio();
    try
    {
        time_t t = time(NULL);
        clock_t c = clock();
        const string DSET_DIR = "/data/datasets/faceScrub";
        const string SCRATCH_DIR = "/tmp/sourabh.d";
        const string DOWNLOAD_DIR = SCRATCH_DIR + "/" + "facescrub_images";
        const string IMGFMT = "jpg";
        mkdir_p(SCRATCH_DIR);
        mkdir_p(DOWNLOAD_DIR);

        curl_global_init(CURL_GLOBAL_DEFAULT);

        download_facescrub(DSET_DIR, DOWNLOAD_DIR, "actors");
        download_facescrub(DSET_DIR, DOWNLOAD_DIR, "actresses");
        
        cout << "TOTAL" << " : " << "time = " << (time(NULL) - t) << " clock = " << float(clock() - c) / CLOCKS_PER_SEC << endl; 

        curl_global_cleanup();

    }
     catch (exception &e)
    {
        cerr << e.what() << endl;
    }

    return 0;
}
