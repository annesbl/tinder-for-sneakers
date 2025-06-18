import numpy as np
import random
import json
from typing import List, Dict, Tuple
from annoy import AnnoyIndex

class ShoeRecommenderWithExistingIndex:
    def __init__(self, component_indices: Dict[str, AnnoyIndex], component_mappings: Dict[str, Dict]):
        """
        Recommender mit separaten Annoy Indices pro Komponente
        
        Args:
            component_indices: {'ganzer_schuh': index1, 'sohle': index2, 'schnuersenkel': index3, 'farbe': index4}
            component_mappings: {'ganzer_schuh': {0: 'shoe_id1', 1: 'shoe_id2', ...}, ...}
        """
        self.indices = component_indices
        self.component_mappings = component_mappings
        
        # Haupt-Mapping vom ganzer_schuh index verwenden
        if 'ganzer_schuh' not in component_mappings:
            raise ValueError("ganzer_schuh mapping required as main reference")
            
        self.shoe_ids = list(component_mappings['ganzer_schuh'].values())
        self.n_items = len(self.shoe_ids)
        
        # Mappings für schnellen Zugriff (basierend auf ganzer_schuh)
        self.shoe_to_idx = {shoe_id: idx for idx, shoe_id in component_mappings['ganzer_schuh'].items()}
        self.idx_to_shoe = component_mappings['ganzer_schuh']
        
        print(f"Loaded recommender with {self.n_items} shoes")
        print(f"Available components: {list(component_indices.keys())}")
    
    def get_diverse_starter_shoes(self, n_shoes: int = 3) -> List[str]:
        """
        3 möglichst unterschiedliche Schuhe für den Anfang
        Nutzt den ganzer_schuh Index für maximale Diversität
        """
        if 'ganzer_schuh' not in self.indices:
            raise ValueError("ganzer_schuh index required for diverse starter selection")
            
        if self.n_items <= n_shoes:
            return self.shoe_ids.copy()
        
        ganzer_schuh_index = self.indices['ganzer_schuh']
        
        # Ersten Schuh random wählen
        selected_indices = [random.randint(0, self.n_items - 1)]
        
        # Weitere Schuhe: maximale Distanz zu bereits gewählten
        for _ in range(n_shoes - 1):
            best_idx = None
            best_min_distance = -1
            
            for candidate_idx in range(self.n_items):
                if candidate_idx in selected_indices:
                    continue
                
                # Minimale Distanz zu allen bereits gewählten
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    # Distanz über nearest neighbors approximieren
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
        Komponenten-Vektor aus dem entsprechenden Annoy Index holen
        
        Args:
            annoy_idx: Index im Annoy
            component: 'ganzer_schuh', 'sohle', 'schnuersenkel', 'farbe'
        
        Returns:
            Komponenten-Vektor
        """
        if component not in self.indices:
            raise ValueError(f"Component '{component}' not available. Available: {list(self.indices.keys())}")
        
        return np.array(self.indices[component].get_item_vector(annoy_idx))
    
    def process_user_feedback(self, annoy_idx: int, feedback: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        User-Feedback verarbeiten
        
        Args:
            annoy_idx: Index des bewerteten Schuhs im Annoy
            feedback: {"ganzer_schuh": 1, "sohle": 0, "schnuersenkel": 1, "farbe": 0}
        
        Returns:
            Preference vectors für gelikte Komponenten
        """
        preference_vectors = {}
        
        for component, liked in feedback.items():
            if liked == 1 and component in self.indices:  # Nur positive Likes und verfügbare Komponenten
                component_vector = self.get_component_vector(annoy_idx, component)
                preference_vectors[component] = component_vector
        
        return preference_vectors
    
    def get_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                          exploration_rate: float = 0.3,
                          exploration_percentile: float = 0.5,
                          shown_indices: set = None) -> List[int]:
        """
        Empfehlungen generieren - gibt nur EINEN Schuh zurück
        
        Args:
            user_preferences: {component: [liked_vectors]}
            exploration_rate: Wahrscheinlichkeit für Exploration (0.3 = 30%)
            exploration_percentile: Aus den besten X% der ähnlichen Schuhe zufällig wählen (0.5 = 50%)
            shown_indices: Bereits gezeigte Annoy-Indices
        
        Returns:
            Liste mit einem Schuh-Index
        """
        if shown_indices is None:
            shown_indices = set()
        
        # Entscheidung: Exploitation vs Exploration
        use_exploration = random.random() >= (1 - exploration_rate)  # 30% Chance für Exploration
        
        if user_preferences and use_exploration:
            # 30% Chance: Exploration - zufällig aus mittleren 50%
            explore_candidates = self._get_smart_exploration_recommendations(
                user_preferences, 1, exploration_percentile, shown_indices, set()  # Nur 1 Schuh!
            )
            if explore_candidates:
                return [explore_candidates[0]]  # Ersten Exploration-Kandidaten nehmen
        
        # 70% Chance: Exploitation - besten ähnlichen Schuh
        if user_preferences:
            exploit_candidates = self._get_preference_based_recommendations(
                user_preferences, 1, shown_indices  # Nur 1 Schuh statt 10!
            )
            if exploit_candidates:
                return [exploit_candidates[0]]  # Besten Exploitation-Kandidaten nehmen
        
        # Fallback: Zufälliger Schuh falls keine Präferenzen
        available_indices = set(range(self.n_items)) - shown_indices
        if available_indices:
            return [random.choice(list(available_indices))]
        
        return []  # Keine Schuhe mehr verfügbar
    
    def _get_preference_based_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                                            n_recommendations: int, 
                                            shown_indices: set) -> List[int]:
        """Gibt nur den BESTEN ähnlichen Schuh zurück"""
        
        # Nur die letzten 5 Likes pro Komponente verwenden
        recent_preferences = self._get_recent_preferences(user_preferences, recent_limit=5)
        
        # Alle gelikten Vektoren zusammenfassen
        all_vectors = []
        for component, liked_vectors in recent_preferences.items():
            if liked_vectors:
                all_vectors.extend(liked_vectors)
        
        if not all_vectors:
            return []
        
        # Durchschnittsvektor aller Likes
        avg_vector = np.mean(all_vectors, axis=0)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        
        # Den einen besten ähnlichen Schuh finden
        similar_indices, distances = self.indices['ganzer_schuh'].get_nns_by_vector(
            avg_vector.tolist(),
            len(shown_indices) + 10,  # Ein paar mehr falls manche schon gezeigt
            include_distances=True
        )
        
        # Ersten noch nicht gezeigten zurückgeben
        for idx in similar_indices:
            if idx not in shown_indices:
                return [idx]
        
        return []
    
    def _get_smart_exploration_recommendations(self, user_preferences: Dict[str, List[np.ndarray]], 
                                             n_recommendations: int,
                                             exploration_percentile: float,
                                             shown_indices: set,
                                             already_recommended: set) -> List[int]:
        """
        Smarte Exploration: Zufällig aus den besten X% ALLER Schuhe wählen
        """
        if not user_preferences:
            return []
        
        # Durchschnitts-Präferenz-Vektor berechnen (über alle Komponenten)
        all_vectors = []
        for component, liked_vectors in user_preferences.items():
            if liked_vectors:
                # Nur die letzten 5 pro Komponente  
                recent_vectors = liked_vectors[-5:]
                all_vectors.extend(recent_vectors)
        
        if not all_vectors:
            return []
        
        # Gesamt-Präferenz-Vektor
        avg_vector = np.mean(all_vectors, axis=0)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        
        # Anzahl Schuhe für die besten X% berechnen
        n_top_candidates = int(self.n_items * exploration_percentile)  # 50% von 900 = 450 Schuhe
        
        # Die besten X% aller Schuhe finden
        similar_indices, distances = self.indices['ganzer_schuh'].get_nns_by_vector(
            avg_vector.tolist(),
            n_top_candidates,  # z.B. 450 beste Schuhe
            include_distances=True
        )
        
        # Bereits gezeigte ausschließen
        available_candidates = []
        for idx in similar_indices:
            if idx not in shown_indices and idx not in already_recommended:
                available_candidates.append(idx)
        
        # Daraus zufällig wählen
        if len(available_candidates) <= n_recommendations:
            return available_candidates
        else:
            return random.sample(available_candidates, n_recommendations)
    
    def _get_recent_preferences(self, user_preferences: Dict[str, List[np.ndarray]], 
                               recent_limit: int = 5) -> Dict[str, List[np.ndarray]]:
        """
        Nur die letzten N Likes pro Komponente verwenden
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
        """Nächsten Schuh-Index holen"""
        
        # Onboarding: Erste 3 diverse Schuhe
        if self.is_onboarding and self.onboarding_count < 3:
            if self.onboarding_count == 0:
                # Diverse Starter beim ersten Aufruf berechnen
                self.starter_shoes = self.recommender.get_diverse_starter_shoes(3)
            
            if self.onboarding_count < len(self.starter_shoes):
                shoe_id = self.starter_shoes[self.onboarding_count]
                shoe_idx = self.recommender.shoe_to_idx[shoe_id]
                self.shown_indices.add(shoe_idx)
                self.onboarding_count += 1
                
                if self.onboarding_count >= 3:
                    self.is_onboarding = False
                
                return shoe_idx
        
        # Nach Onboarding: Empfehlungen (gibt nur 1 Schuh zurück)
        recommendations = self.recommender.get_recommendations(
            self.user_preferences,
            exploration_rate=0.3,
            exploration_percentile=0.5,  # Aus besten 50% der ähnlichen Schuhe
            shown_indices=self.shown_indices
        )
        
        if recommendations:
            next_idx = recommendations[0]
            self.shown_indices.add(next_idx)
            return next_idx
        
        return None  # Keine weiteren Schuhe
    
    def submit_feedback(self, annoy_idx: int, feedback: Dict[str, int]):
        """
        User-Feedback verarbeiten
        
        Args:
            annoy_idx: Index des bewerteten Schuhs
            feedback: {"ganzer_schuh": 1, "sohle": 0, "schnuersenkel": 1, "farbe": 0}
        """
        preference_vectors = self.recommender.process_user_feedback(annoy_idx, feedback)
        
        # Zu Präferenzen hinzufügen
        for component, vector in preference_vectors.items():
            self.user_preferences[component].append(vector)
        
        print(f"Feedback for index {annoy_idx}: {feedback}")
        liked_components = [comp for comp, vectors in self.user_preferences.items() if vectors]
        print(f"User now has preferences for: {liked_components}")
        
        # Debug: Zeige nur die letzten 5 pro Komponente
        for component, vectors in self.user_preferences.items():
            if len(vectors) > 5:
                print(f"  {component}: {len(vectors)} total likes, using last 5 for recommendations")
        
        print(f"Next recommendation will be: {'EXPLORATION' if random.random() < 0.3 else 'EXPLOITATION'} (preview)")

    
    def get_next_shoe_id(self) -> str:
        """Convenience method um shoe_id statt index zu bekommen"""
        idx = self.get_next_shoe_index()
        if idx is not None:
            return self.recommender.idx_to_shoe[idx]
        return None


# INITIALISIERUNG - Hier musst du deine Annoy-Files laden
def load_recommender():
    """Lädt alle Annoy-Indizes und Mappings"""
    
    # Annoy-Indizes laden
    ganzer_schuh_index = AnnoyIndex(512, 'angular')
    ganzer_schuh_index.load('sneakers_ganzer_schuh.ann')
    
    sohle_index = AnnoyIndex(512, 'angular')
    sohle_index.load('sneakers_sohle.ann')
    
    schnuersenkel_index = AnnoyIndex(512, 'angular')
    schnuersenkel_index.load('sneakers_schnuersenkel.ann')
    
    farbe_index = AnnoyIndex(512, 'angular')
    farbe_index.load('sneakers_farbe.ann')
    
    # Mappings laden
    with open('sneakers_ganzer_schuh_mapping.json', 'r') as f:
        ganzer_schuh_mapping = json.load(f)
        # String-Keys zu Int konvertieren
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
    
    # Zusammenpacken
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


# # BEISPIEL VERWENDUNG:
# if __name__ == "__main__":
#     # Recommender laden
#     recommender = load_recommender()
    
#     # Session starten
#     session = ShoeSession(recommender)
    
#     # Ersten Schuh holen
#     shoe_idx = session.get_next_shoe_index()
#     print(f"Zeige Schuh mit Index: {shoe_idx}")
    
#     # Feedback simulieren
#     feedback = {"ganzer_schuh": 1, "sohle": 0, "schnuersenkel": 1, "farbe": 0}
#     session.submit_feedback(shoe_idx, feedback)
    
#     # Nächsten Schuh holen
#     next_shoe_idx = session.get_next_shoe_index()
#     print(f"Nächster Schuh Index: {next_shoe_idx}")