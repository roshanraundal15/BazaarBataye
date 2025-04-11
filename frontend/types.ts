export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  public: {
    Tables: {
      commodities: {
        Row: {
          base_price: number
          created_at: string
          icon: string | null
          id: string
          name: string
          price_variability: number
          rainfall_sensitivity: number
        }
        Insert: {
          base_price: number
          created_at?: string
          icon?: string | null
          id: string
          name: string
          price_variability: number
          rainfall_sensitivity: number
        }
        Update: {
          base_price?: number
          created_at?: string
          icon?: string | null
          id?: string
          name?: string
          price_variability?: number
          rainfall_sensitivity?: number
        }
        Relationships: []
      }
      districts: {
        Row: {
          created_at: string
          id: string
          name: string
          state: string
        }
        Insert: {
          created_at?: string
          id: string
          name: string
          state: string
        }
        Update: {
          created_at?: string
          id?: string
          name?: string
          state?: string
        }
        Relationships: []
      }
      historical_prices: {
        Row: {
          commodity_id: string
          created_at: string
          date: string
          id: string
          price: number
          volume: number | null
        }
        Insert: {
          commodity_id: string
          created_at?: string
          date: string
          id?: string
          price: number
          volume?: number | null
        }
        Update: {
          commodity_id?: string
          created_at?: string
          date?: string
          id?: string
          price?: number
          volume?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "historical_prices_commodity_id_fkey"
            columns: ["commodity_id"]
            isOneToOne: false
            referencedRelation: "commodities"
            referencedColumns: ["id"]
          },
        ]
      }
      price_predictions: {
        Row: {
          commodity_id: string
          confidence: number
          created_at: string
          date: string
          district_id: string
          id: string
          lower_bound: number
          predicted_price: number
          upper_bound: number
        }
        Insert: {
          commodity_id: string
          confidence: number
          created_at?: string
          date: string
          district_id: string
          id?: string
          lower_bound: number
          predicted_price: number
          upper_bound: number
        }
        Update: {
          commodity_id?: string
          confidence?: number
          created_at?: string
          date?: string
          district_id?: string
          id?: string
          lower_bound?: number
          predicted_price?: number
          upper_bound?: number
        }
        Relationships: [
          {
            foreignKeyName: "price_predictions_commodity_id_fkey"
            columns: ["commodity_id"]
            isOneToOne: false
            referencedRelation: "commodities"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "price_predictions_district_id_fkey"
            columns: ["district_id"]
            isOneToOne: false
            referencedRelation: "districts"
            referencedColumns: ["id"]
          },
        ]
      }
      rainfall_data: {
        Row: {
          created_at: string
          date: string
          district_id: string
          id: string
          rainfall_mm: number
        }
        Insert: {
          created_at?: string
          date: string
          district_id: string
          id?: string
          rainfall_mm: number
        }
        Update: {
          created_at?: string
          date?: string
          district_id?: string
          id?: string
          rainfall_mm?: number
        }
        Relationships: [
          {
            foreignKeyName: "rainfall_data_district_id_fkey"
            columns: ["district_id"]
            isOneToOne: false
            referencedRelation: "districts"
            referencedColumns: ["id"]
          },
        ]
      }
      trained_models: {
        Row: {
          accuracy: number
          active: boolean | null
          commodity_id: string
          district_id: string
          id: string
          model_parameters: Json
          training_date: string
        }
        Insert: {
          accuracy: number
          active?: boolean | null
          commodity_id: string
          district_id: string
          id?: string
          model_parameters: Json
          training_date?: string
        }
        Update: {
          accuracy?: number
          active?: boolean | null
          commodity_id?: string
          district_id?: string
          id?: string
          model_parameters?: Json
          training_date?: string
        }
        Relationships: [
          {
            foreignKeyName: "trained_models_commodity_id_fkey"
            columns: ["commodity_id"]
            isOneToOne: false
            referencedRelation: "commodities"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "trained_models_district_id_fkey"
            columns: ["district_id"]
            isOneToOne: false
            referencedRelation: "districts"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      seed_initial_sample_data: {
        Args: Record<PropertyKey, never>
        Returns: undefined
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DefaultSchema = Database[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof Database },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof Database },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends { schema: keyof Database }
  ? Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
