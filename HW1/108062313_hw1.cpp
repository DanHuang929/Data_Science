#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>
#include <iomanip>
using namespace std;


void apriori();
vector<int> string_to_num(string);
vector<vector<int> > first_join();
float calculate_support(vector<int>);
vector<vector<int> > join(vector<vector<int> >, int);
vector<vector<int> > generate_L(vector<vector<int> >);
vector<vector<int> > generate_C(vector<vector<int> >, int);
vector<vector<int> > reduce (vector<vector<int> >, vector<vector<int> >);

float min_support;
vector<float> support;
vector<vector<int> > transaction;
vector<vector<int> > frequent_pattern;


vector<int> string_to_num(string s)
{
	bool temp=false;		
	int data=0;				
	vector<int> arr;				

	for(int i=0;i<s.length();i++){
		while((s[i]>='0')&&(s[i]<='9')){
			temp=true;		
			data*=10;
			data+=(s[i]-'0');		
			i++;
		}

		if(temp){
			arr.push_back(data);	
			data=0;		
			temp=false;	
		}
	}
	sort(arr.begin(), arr.end());
	return arr;
}


void apriori() {
	int step=0;
	vector<vector<int> > C, L;
	while(true) {
		C = generate_C(L, step);
		if(C.size()==0) break;
		step++;
		L = generate_L(C);
	
	}
}


vector<vector<int> > generate_C(vector<vector<int> > L, int step) {
	vector<vector<int> > C, temp_C;
	if(step==0) 
        C = first_join();
	else{
		temp_C = join(L, step);
		C = reduce(L,temp_C);
	}
	return C;
}

vector<vector<int> > first_join(){
	vector<vector<int> > C;
	set<int> temp;
	for (int i=0; i<transaction.size(); i++)
		for (int j=0; j<transaction[i].size(); j++)
			temp.insert(transaction[i][j]);
	for(auto iter=temp.begin(); iter != temp.end(); iter++){
		vector<int> new_C;
		new_C.push_back(*iter);
		C.push_back(new_C);
	}
	return C;
}

vector<vector<int> > join(vector<vector<int> > L, int step) {
	vector<vector<int> > C;
	int len=step-1;
	for(int i=0;i<L.size();i++){
		for(int j=i+1;j<L.size(); j++) {
			int same=1;
			for(int k=0; k<len; k++) {
				if(L[i][k] != L[j][k]){
					same=0;
					break;
				}
			}
			if(same) { //new C
				vector<int> new_C;
				for(int z=0;z<len; z++) {
					new_C.push_back(L[i][z]);
				}
				int n = L[i][len];
				int m = L[j][len];

				if(n>m){
					new_C.push_back(m);
					new_C.push_back(n);
				}
				else{
					new_C.push_back(n);
					new_C.push_back(m);
				}
				C.push_back(new_C);
			}
		}
	}
	return C;
}

vector<vector<int> > reduce (vector<vector<int> > L, vector<vector<int> > C) {
        vector<vector<int> > new_C;
        set<vector<int> > L_set;
        for(int i=0;i<L.size();i++)
			L_set.insert(L[i]);
        
        for(int i=0;i<C.size();i++){
            int flag=1;
            for(int j=0;j<C[i].size();j++){
                vector<int> tmp = C[i];
                tmp.erase(tmp.begin()+j);
                if(L_set.find(tmp) == L_set.end()) { //not find
					flag=0;
                    break;
                }
            }
            if(flag){
                new_C.push_back(C[i]);
            }
        }
        return new_C;
    }

float calculate_support(vector<int> item) {
	int ret = 0;
	for(int k=0; k<transaction.size(); k++){
		if(transaction[k].size() < item.size()) continue;

		int i, j;
		for(i=0, j=0; i < transaction[k].size(); i++) {
			if(transaction[k][i]==item[j]) j++;
			if(j==item.size()) break;
		}
		if(j==item.size()){ //all same
			ret++;
		}
	}
	return (float)ret/transaction.size();
}

vector<vector<int> > generate_L(vector<vector<int> > C) {
	vector<vector<int> > L;
	for(int i=0; i<C.size(); i++){
		float S = calculate_support(C[i]);
		// cout<<S<<" ###"<<endl;
		if(S < min_support) continue;
		else{
			L.push_back(C[i]);
			frequent_pattern.push_back(C[i]);
			support.push_back(S);
		}
		
	}
	return L;
}

int main(int argc, char *argv[]) {
	
	min_support = stod(argv[1]);
	string input_file = argv[2];
	string output_file = argv[3];
      
    string s;	
	ifstream ifs;
	ifs.open(input_file);
	while(getline(ifs, s)){
        transaction.push_back(string_to_num(s));
	}
	ifs.close();
			

	apriori();

	ofstream ofs;
	ofs.open(output_file);
	for(int i=0; i<frequent_pattern.size(); i++){
        for(int j=0; j<frequent_pattern[i].size()-1; j++){
			ofs<<frequent_pattern[i][j]<<",";
        }
		ofs<<frequent_pattern[i][frequent_pattern[i].size()-1]<<":";
		ofs<<fixed<<setprecision(4)<<support[i]<<endl;
    }
	ofs.close();	

    return 0;
}