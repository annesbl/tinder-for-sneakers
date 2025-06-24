import numpy as np
import random
import json
from typing import List, Dict, Tuple
from annoy import AnnoyIndex

class ShoeRecommenderWithExistingIndex:
    def __init__(self, component_indices: Dict[str, AnnoyIndex], component_mappings: Dict[str, Dict]):
        """
        Args:
            component_indices: {'ganzer_schuh': index1, 'sohle': index2, 'schnuersenkel': index3, 'farbe': index4}
            component_mappings: {'ganzer_schuh': {0: 'shoe_id1', 1: 'shoe_id2', ...}, ...}
        """
        self.indices = component_indices
        self.component_mappings = component_mappings
        
        # mapping of whole shoe to indices
        if 'ganzer_schuh' not in component_mappings:
            raise ValueError("ganzer_schuh mapping required as main reference")
            
        self.shoe_ids = list(component_mappings['ganzer_schuh'].values())
        self.n_items = len(self.shoe_ids)
        
        # Mappings for fast access
        self.shoe_to_idx = {shoe_id: idx for idx, shoe_id in component_mappings['ganzer_schuh'].items()}
        self.idx_to_shoe = component_mappings['ganzer_schuh']
        
        print(f"Loaded recommender with {self.n_items} shoes")
        print(f"Available components: {list(component_indices.keys())}")
    
    def get_diverse_starter_shoes(self, n_shoes: int = 3) -> List[str]:
        """
        Select diverse starter shoes that are as different as possible.
        Solves the cold-start problem by maximizing diversity of initial recommendations.
        
        Args:
            n_shoes: Number of starter shoes to select
        
        Returns:
            List of shoe IDs for the selected diverse starter shoes
        """
        if 'ganzer_schuh' not in self.indices:
            raise ValueError("ganzer_schuh index required for diverse starter selection")
            
        if self.n_items <= n_shoes:
            return self.shoe_ids.copy()
                
        # pick random first shoe
        selected_indices = [random.randint(0, self.n_items - 1)]
        
        # more shoes: max distance to already selected
        for _ in range(n_shoes - 1):
            best_idx = None
            best_min_distance = -1
            
            for candidate_idx in range(self.n_items):
                if candidate_idx in selected_indices:
                    continue
                
                # minimal distance to already selected
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    # get nearest neighbors
                    neighbors, distances = self.indices['ganzer_schuh'].get_nns_by_item(
                        candidate_idx, min(100, self.n_items), include_distances=True
                    )
                    
                    if selected_idx in neighbors:
                        distance = distances[neighbors.index(selected_idx)]
                        min_distance = min(min_distance, distance)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
        
        return [self.shoe_ids[idx] for idx in selected_indices]
    
    def get_component_vector(self, annoy_idx: int, component: str) -> np.ndarray:
        """
        get component vector from Annoy
        
        Args:
            annoy_idx: index in annoy
            component: 'ganzer_schuh', 'sohle', 'schnuersenkel', 'farbe'
        
        Returns:
            component vector
        """
        if component not in self.indices:
            raise ValueError(f"Component '{component}' not available. Available: {list(self.indices.keys())}")
        
        return np.array(self.indices[component].get_item_vector(annoy_idx))
    
    def process_user_feedback(self, annoy_idx: int, feedback: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        processing of user feedback
        
        Args:
            annoy_idx:  index of reviewed shoe
            feedback: {"ganzer_schuh": 1, "sohle": 0, "schnuersenkel": 1, "farbe": 0}
        
        Returns:
            Preference vectors for liked components
        """
        preference_vectors = {}
        
        for component, liked in feedback.items():
            if liked == 1 and component in self.indices:  
                component_vector = self.get_component_vector(annoy_idx, component)
                preference_vectors[component] = component_vector
        
        return preference_vectors
    
    def get_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                          exploration_rate: float = 0.3,
                          exploration_percentile: float = 0.5,
                          shown_indices: set = None) -> List[int]:
        """
        returns shoe which will be recommended
        
        Args:
            user_preferences: {component: [liked_vectors]}
            exploration_rate: probability for exploration (0.3 = 30%)
            exploration_percentile: from best X% pick a random shoe (0.5 = 50%)
            shown_indices: already shown shoes
        
        Returns:
            list with a shoe id
        """
        if shown_indices is None:
            shown_indices = set()
        
        # Exploitation or Exploration
        use_exploration = random.random() >= (1 - exploration_rate)  # 30% chance für Exploration
        
        if user_preferences and use_exploration:
            # 30% chance: Exploration - random from top X%
            explore_candidates = self._get_smart_exploration_recommendations(
                user_preferences, 1, exploration_percentile, shown_indices  
            )
            if explore_candidates:
                return [explore_candidates[0]]  
        
        # 70% Chance: Exploitation - most similar shoe
        if user_preferences:
            exploit_candidates = self._get_preference_based_recommendations(
                user_preferences, 1, shown_indices  
            )
            if exploit_candidates:
                return [exploit_candidates[0]] 
        
        # random shoe if no user preferences
        available_indices = set(range(self.n_items)) - shown_indices
        if available_indices:
            return [random.choice(list(available_indices))]
        
        return []  
    
    def _get_preference_based_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                                            n_recommendations: int, 
                                            shown_indices: set) -> List[int]:
        """
        Get recommendations based on user preferences.
        Finds shoes most similar to the user's averaged preference vectors.
        
        Args:
            user_preferences: Dictionary mapping component names to lists of liked embedding vectors
            n_recommendations: Number of recommendations to return
            shown_indices: Set of already shown shoe indices to exclude
        
        Returns:
            shoe index of shoe to recommend
        """
        
        # only use likes from last 5
        recent_preferences = self._get_recent_preferences(user_preferences, recent_limit=5)
        
        # combine all liked vectors
        all_vectors = []
        for component, liked_vectors in recent_preferences.items():
            if liked_vectors:
                all_vectors.extend(liked_vectors)
        
        if not all_vectors:
            return []
        
        # calculate average preference vector
        avg_vector = np.mean(all_vectors, axis=0)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        
        # find most similar shoe
        similar_indices, distances = self.indices['ganzer_schuh'].get_nns_by_vector(
            avg_vector.tolist(),
            len(shown_indices) + 10,  # return more shoes if best match is already shown
            include_distances=True
        )
        
        # check if similar shoe is not already shown
        for idx in similar_indices:
            if idx not in shown_indices:
                return [idx]
        
        return []
    
    def _get_smart_exploration_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                                             n_recommendations: int,
                                             exploration_percentile: float,
                                             shown_indices: set) -> List[int]:
        """
        Get recommendations, based on user preferences, but not the best matching shoe.
        Random shoe from top X% of shoes similar to user preferences.
        
        Args:
            user_preferences: Dictionary mapping component names to lists of liked embedding vectors
            n_recommendations: Number of recommendations to return
            exploration_percentile: Percentile of top similar shoes to sample from (0.5 = top 50%)
            shown_indices: Set of already shown shoe indices to exclude
            already_recommended: Set of already recommended shoe indices to exclude
        
        Returns:
            List of recommended shoe indices from smart exploration
        """
        if not user_preferences:
            return []
        
        # calculate average preference vector
        all_vectors = []
        for component, liked_vectors in user_preferences.items():
            if liked_vectors:
                # only use recent likes (last 5)  
                recent_vectors = liked_vectors[-5:]
                all_vectors.extend(recent_vectors)
        
        if not all_vectors:
            return []
        
        # whole shoe vector
        avg_vector = np.mean(all_vectors, axis=0)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        
        # calculate number of top candidates
        n_top_candidates = int(self.n_items * exploration_percentile)  # 50% von 900 = 450 Schuhe
        
        # find best X% similar shoes
        similar_indices, distances = self.indices['ganzer_schuh'].get_nns_by_vector(
            avg_vector.tolist(),
            n_top_candidates,  # z.B. 450 beste Schuhe
            include_distances=True
        )
        
        # do not recommend already shown or recommended shoes
        available_candidates = []
        for idx in similar_indices:
            if idx not in shown_indices:
                available_candidates.append(idx)
        
        # get random shoe from top X%
        return random.sample(available_candidates, n_recommendations)
    
    def _get_recent_preferences(self, user_preferences: Dict[str, List[np.ndarray]], 
                               recent_limit: int = 3) -> Dict[str, List[np.ndarray]]:
        """
        Extract only the most recent preferences to avoid preference drift.
        Keeps only the last N liked vectors per component.
        
        Args:
            user_preferences: Dictionary mapping component names to lists of liked embedding vectors
            recent_limit: Maximum number of recent preferences to keep per component
        
        Returns:
            Dictionary with the same structure but containing only recent preferences
        """
        recent_prefs = {}
        for component, vectors in user_preferences.items():
            if vectors:
                recent_prefs[component] = vectors[-recent_limit:]
            else:
                recent_prefs[component] = []
        return recent_prefs

# Session-Klasse für einfache Nutzung
class ShoeSession:
    def __init__(self, recommender: ShoeRecommenderWithExistingIndex):
        self.recommender = recommender
        self.user_preferences = {'ganzer_schuh': [], 'sohle': [], 'schnuersenkel': [], 'farbe': []}
        self.shown_indices = set()
        self.is_onboarding = True
        self.onboarding_count = 0
        
    def get_next_shoe_index(self) -> int:
        """
        Get the next shoe index to recommend based on current session state.
        Handles onboarding phase with diverse starters, then switches to preference-based recommendations.
        
        Returns:
            Annoy index of the next shoe to recommend, or None if no more shoes available
        """
        
        # Onboarding: first: 3 diverse shoes
        if self.is_onboarding and self.onboarding_count < 3:
            if self.onboarding_count == 0:
                # 3 shoes as different as possible
                self.starter_shoes = self.recommender.get_diverse_starter_shoes(3)
            
            if self.onboarding_count < len(self.starter_shoes):
                shoe_id = self.starter_shoes[self.onboarding_count]
                shoe_idx = self.recommender.shoe_to_idx[shoe_id]
                self.shown_indices.add(shoe_idx)
                self.onboarding_count += 1
                
                if self.onboarding_count >= 3:
                    self.is_onboarding = False
                
                return shoe_idx
        
        # after onboarding, smart exploration
        recommendations = self.recommender.get_recommendations(
            self.user_preferences,
            exploration_rate=0.3,
            exploration_percentile=0.5,  # random shoe from top 50%
            shown_indices=self.shown_indices
        )
        
        if recommendations:
            next_idx = recommendations[0]
            self.shown_indices.add(next_idx)
            return next_idx
        
        return None 
    
    def submit_feedback(self, annoy_idx: int, feedback: Dict[str, int]):
        """
        Process and store user feedback for a rated shoe.
        Updates user preferences based on liked components.
        
        Args:
            annoy_idx: Index of the rated shoe
            feedback: Dictionary mapping component names to ratings (0=dislike, 1=like)
                    {"ganzer_schuh": 1, "sohle": 0, "schnuersenkel": 1, "farbe": 0}
        """
        preference_vectors = self.recommender.process_user_feedback(annoy_idx, feedback)
        
        # add to user preferences
        for component, vector in preference_vectors.items():
            self.user_preferences[component].append(vector)
        
    
    def get_next_shoe_id(self) -> str:
        """
        Convenience method to get the next shoe ID instead of index.
        
        Returns:
            Shoe ID string of the next recommendation, or None if no more shoes available
        """
        idx = self.get_next_shoe_index()
        if idx is not None:
            return self.recommender.idx_to_shoe[idx]
        return None


def load_recommender():
    """
    Load and initialize the shoe recommender with all Annoy indices and mappings.
    Loads pre-built indices and mappings from disk files.
    
    Returns:
        Initialized ShoeRecommenderWithExistingIndex instance ready for use
    """
    
    # get annoy indices
    ganzer_schuh_index = AnnoyIndex(512, 'angular')
    ganzer_schuh_index.load('sneakers_ganzer_schuh.ann')
    
    sohle_index = AnnoyIndex(512, 'angular')
    sohle_index.load('sneakers_sohle.ann')
    
    schnuersenkel_index = AnnoyIndex(512, 'angular')
    schnuersenkel_index.load('sneakers_schnuersenkel.ann')
    
    farbe_index = AnnoyIndex(512, 'angular')
    farbe_index.load('sneakers_farbe.ann')
    
    # get mappings, convert key to int
    with open('sneakers_ganzer_schuh_mapping.json', 'r') as f:
        ganzer_schuh_mapping = json.load(f)
        ganzer_schuh_mapping = {int(k): v for k, v in ganzer_schuh_mapping.items()}
    
    with open('sneakers_sohle_mapping.json', 'r') as f:
        sohle_mapping = json.load(f)
        sohle_mapping = {int(k): v for k, v in sohle_mapping.items()}
    
    with open('sneakers_schnuersenkel_mapping.json', 'r') as f:
        schnuersenkel_mapping = json.load(f)
        schnuersenkel_mapping = {int(k): v for k, v in schnuersenkel_mapping.items()}
    
    with open('sneakers_farbe_mapping.json', 'r') as f:
        farbe_mapping = json.load(f)
        farbe_mapping = {int(k): v for k, v in farbe_mapping.items()}
    
    component_indices = {
        'ganzer_schuh': ganzer_schuh_index,
        'sohle': sohle_index,
        'schnuersenkel': schnuersenkel_index,
        'farbe': farbe_index
    }
    
    component_mappings = {
        'ganzer_schuh': ganzer_schuh_mapping,
        'sohle': sohle_mapping,
        'schnuersenkel': schnuersenkel_mapping,
        'farbe': farbe_mapping
    }
    
    return ShoeRecommenderWithExistingIndex(component_indices, component_mappings)


