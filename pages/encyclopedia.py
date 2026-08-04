"""
Mushroom Encyclopedia Page - Educational content about mushroom features,
their values, and why they matter for identification.
"""
import streamlit as st
import pandas as pd

from core.model_loader import get_feature_options
from components.cards import info_card
from config import FEATURE_DISPLAY_NAMES, FEATURE_DESCRIPTIONS


def app():
    """Render the Mushroom Encyclopedia page."""
    
    st.markdown(
        """
        <div class="fade-in">
            <h1 style="margin-bottom: 0.5rem;">📖 Mushroom Encyclopedia</h1>
            <p style="color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 1.5rem;">
                Learn about the physical characteristics used for mushroom classification.
                Understanding these features helps you identify mushrooms in the wild.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick Navigation ───────────────────────────────────────────
    st.markdown("### 🔍 Quick Navigation")
    st.markdown("Select a feature to learn more about it:")
    
    # Load feature options
    feature_options = get_feature_options()
    
    if not feature_options:
        st.warning("Feature data not available. Please ensure the model files are loaded correctly.")
        return
    
    # Feature selector
    feature_names = list(FEATURE_DISPLAY_NAMES.keys())
    selected_feature = st.selectbox(
        "Choose a feature",
        options=feature_names,
        format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x.replace("-", " ").title()),
        key="encyclopedia_feature_select",
    )
    
    if selected_feature:
        display_name = FEATURE_DISPLAY_NAMES.get(selected_feature, selected_feature.replace("-", " ").title())
        description = FEATURE_DESCRIPTIONS.get(selected_feature, "No description available.")
        options = feature_options.get(selected_feature, [])
        
        # ── Feature Detail Card ────────────────────────────────────
        st.markdown("---")
        
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            st.markdown(
                f"""
                <div class="card">
                    <h2 style="margin-bottom: 0.5rem;">{display_name}</h2>
                    <p style="font-size: 1.05rem; line-height: 1.6; color: var(--text-secondary);">
                        {description}
                    </p>
                    <p style="margin-top: 1rem;">
                        <strong>Feature Key:</strong> <code>{selected_feature}</code>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # ── Feature Values ────────────────────────────────────
            st.markdown("#### Possible Values")
            
            if options:
                # Create a nice display of values
                cols = st.columns(3)
                for i, option in enumerate(options):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div class="card" style="text-align: center; padding: 0.75rem;">
                                <div style="font-weight: 600; font-size: 1.1rem;">{option}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No predefined values available for this feature.")
        
        with col_f2:
            # ── Importance Indicator ──────────────────────────────
            st.markdown(
                """
                <div class="card" style="text-align: center;">
                    <h4>Feature Role</h4>
                    <div style="font-size: 3rem; margin: 1rem 0;">🔑</div>
                    <p style="font-size: 0.9rem; color: var(--text-secondary);">
                        This feature is used as one of the 22 inputs to the SVM model for 
                        predicting mushroom edibility.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # ── Tip ───────────────────────────────────────────────
            st.markdown(
                """
                <div class="card" style="background: #E8F5E9; border: 1px solid #C8E6C9;">
                    <h4>💡 Identification Tip</h4>
                    <p style="font-size: 0.85rem; color: #2E7D32;">
                        When identifying mushrooms in the wild, always examine multiple features. 
                        No single characteristic is sufficient for accurate identification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.markdown("---")
    
    # ── All Features Table ─────────────────────────────────────────
    st.markdown("### 📋 Complete Feature Reference")
    st.markdown(
        "Below is a comprehensive table of all 22 features used by the classification model:"
    )
    
    # Build feature table
    feature_table_data = []
    for i, (feat_key, feat_name) in enumerate(FEATURE_DISPLAY_NAMES.items(), 1):
        options = feature_options.get(feat_key, [])
        options_str = ", ".join(options[:5])
        if len(options) > 5:
            options_str += f" (+{len(options) - 5} more)"
        
        feature_table_data.append({
            "#": i,
            "Feature": feat_name,
            "Values": options_str if options_str else "N/A",
            "Description": FEATURE_DESCRIPTIONS.get(feat_key, "")[:100] + "...",
        })
    
    feature_df = pd.DataFrame(feature_table_data)
    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Feature": st.column_config.TextColumn("Feature Name", width="medium"),
            "Values": st.column_config.TextColumn("Possible Values", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )
    
    st.markdown("---")
    
    # ── Educational Content ─────────────────────────────────────────
    st.markdown("### 🎓 Mushroom Identification Guide")
    
    tabs = st.tabs([
        "🔬 Getting Started",
        "📏 Key Features",
        "⚠️ Safety Tips",
        "📚 Resources",
    ])
    
    with tabs[0]:
        st.markdown(
            """
            <div class="card">
                <h3>Getting Started with Mushroom Identification</h3>
                <p>
                    Mushroom identification is a fascinating but complex skill. Here are some 
                    tips for beginners:
                </p>
                <ol>
                    <li><strong>Start with the basics:</strong> Learn the major parts of a mushroom 
                        (cap, gills, stalk, ring, volva).</li>
                    <li><strong>Use multiple features:</strong> Never rely on a single characteristic 
                        for identification. Cross-reference multiple features.</li>
                    <li><strong>Take spore prints:</strong> Spore print color is one of the most 
                        reliable identification features.</li>
                    <li><strong>Observe the habitat:</strong> Where a mushroom grows provides 
                        crucial clues about its identity.</li>
                    <li><strong>Note the season:</strong> Many mushrooms fruit only during specific 
                        seasons or weather conditions.</li>
                    <li><strong>Document everything:</strong> Take photos, notes, and sketches 
                        of your findings.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with tabs[1]:
        st.markdown(
            """
            <div class="card">
                <h3>Key Features for Identification</h3>
                <p>
                    Some features are more important for classification than others. Based on 
                    the SHAP analysis, here are the most influential features:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        important_features = [
            (
                "👃",
                "Odor",
                "The smell of a mushroom is the strongest indicator of edibility. "
                "Foul, fishy, or pungent odors often indicate toxicity.",
            ),
            (
                "🎨",
                "Gill Color",
                "Gill color, especially spore print color, is a highly reliable "
                "taxonomic feature for distinguishing between species.",
            ),
            (
                "🔴",
                "Bruises",
                "How a mushroom reacts to bruising (color change) can indicate "
                "the presence of certain compounds and potential toxicity.",
            ),
            (
                "💍",
                "Ring Type",
                "The structure of the ring on the stalk is an important "
                "morphological feature for species identification.",
            ),
            (
                "🎭",
                "Cap Shape",
                "Cap morphology provides essential clues about the mushroom's "
                "genus and species.",
            ),
        ]
        
        for icon, name, desc in important_features:
            st.markdown(
                f"""
                <div class="card" style="border-left: 4px solid var(--primary);">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">{icon}</div>
                        <div>
                            <strong style="font-size: 1.1rem;">{name}</strong>
                            <p style="margin: 0.25rem 0 0 0; color: var(--text-secondary);">{desc}</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    with tabs[2]:
        st.markdown(
            """
            <div class="card" style="background: #FFF3E0; border-left: 4px solid #FF9800;">
                <h3 style="color: #E65100;">⚠️ Mushroom Safety Tips</h3>
                <div style="color: #BF360C;">
                    <p><strong>Never eat a wild mushroom unless you are 100% certain of its identity.</strong></p>
                    <ul>
                        <li>Many poisonous mushrooms closely resemble edible varieties.</li>
                        <li>Cooking does not neutralize all toxins.</li>
                        <li>Some toxins can cause delayed symptoms (up to 24 hours).</li>
                        <li>Children and pets are particularly vulnerable to mushroom poisoning.</li>
                        <li>Always consult with a certified mycologist for verification.</li>
                        <li>When in doubt, throw it out!</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="card">
                <h3>Golden Rules of Foraging</h3>
                <ol>
                    <li><strong>Identify before you pick:</strong> Be confident in your identification 
                        before harvesting any mushroom.</li>
                    <li><strong>Check multiple sources:</strong> Use field guides, apps, and 
                        expert opinions to verify your identification.</li>
                    <li><strong>Start with easy species:</strong> Begin with easily identifiable 
                        species like morels, chanterelles, and puffballs.</li>
                    <li><strong>Join a mycological society:</strong> Local experts can provide 
                        invaluable guidance and verification.</li>
                    <li><strong>Keep a small sample:</strong> When trying a new edible species, 
                        consume only a small amount initially.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with tabs[3]:
        st.markdown(
            """
            <div class="card">
                <h3>Recommended Resources</h3>
                <div style="display: grid; gap: 1rem;">
                    <div class="card" style="padding: 1rem;">
                        <strong>📚 Field Guides</strong>
                        <ul>
                            <li>National Audubon Society Field Guide to Mushrooms</li>
                            <li>Mushrooms Demystified by David Arora</li>
                            <li>All That the Rain Promises and More by David Arora</li>
                        </ul>
                    </div>
                    <div class="card" style="padding: 1rem;">
                        <strong>🌐 Online Resources</strong>
                        <ul>
                            <li>MushroomExpert.com - Comprehensive identification guides</li>
                            <li>iNaturalist.org - Community-based species identification</li>
                            <li>MushroomObserver.org - Citizen science platform</li>
                            <li>North American Mycological Association (NAMA)</li>
                        </ul>
                    </div>
                    <div class="card" style="padding: 1rem;">
                        <strong>📱 Mobile Apps</strong>
                        <ul>
                            <li>iNaturalist - AI-assisted species identification</li>
                            <li>Mushroom Identificator - Photo-based identification</li>
                            <li>ShroomID - Community-verified identification</li>
                        </ul>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # ── Disclaimer ─────────────────────────────────────────────────
    st.markdown("---")
    info_card(
        "⚠️ Important Disclaimer",
        "This encyclopedia is for <strong>educational purposes only</strong>. The information "
        "provided should not be used as the sole basis for identifying or consuming wild mushrooms. "
        "Always consult with qualified experts and multiple reliable sources before making any "
        "decisions about mushroom edibility.",
        icon="⚠️",
        color="#FF9800",
    )
